//! CUDA driver
//!
//! Reference: http://docs.nvidia.com/cuda/cuda-driver-api/

use std;
use std::ffi::CString;
use std::marker::PhantomData;
use std::rc::Rc;
use std::cell::Cell;
use std::{mem, ptr, result};

#[allow(dead_code)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
mod ll;

/// A CUDA "block"
pub struct Block {
    x: u32,
    y: u32,
    z: u32,
}

impl Block {
    /// One dimensional block
    pub fn x(x: u32) -> Self {
        Block { x: x, y: 1, z: 1 }
    }

    /// Two dimensional block
    pub fn xy(x: u32, y: u32) -> Self {
        Block { x: x, y: y, z: 1 }
    }

    /// Three dimensional block
    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Block { x: x, y: y, z: z }
    }
}

/// A CUDA "context"
#[derive(Debug, Clone)]
pub struct Context(Rc<ContextInner>);
#[derive(Debug)]
struct ContextInner {
    poisoned: Cell<bool>,
    handle: ll::CUcontext,
}

impl Context {
    // TODO is this actually useful? Note that we are using "RAII" (cf. `drop`)
    // and ownership to manage `Context`es
    #[allow(dead_code)]
    fn current() -> Result<Option<Self>> {
        let mut handle = ptr::null_mut();

        unsafe { lift(ll::cuCtxGetCurrent(&mut handle))? }

        if handle.is_null() {
            Ok(None)
        } else {
            Ok(Some(Context(Rc::new(ContextInner {
                poisoned: Cell::new(false),
                handle: handle,
            }))))
        }
    }

    /// Binds context to the calling thread
    pub fn set_current(&self) -> Result<()> {
        unsafe { lift(ll::cuCtxSetCurrent(self.0.handle)) }
    }

    /// Loads a PTX module
    pub fn load_module<T: AsRef<str>>(&self, image: T) -> Result<Module> {
        let mut handle = ptr::null_mut();
        let image = CString::new(image.as_ref()).unwrap();

        unsafe {
            lift(ll::cuModuleLoadData(
                &mut handle,
                image.as_ptr() as *const _,
            ))?
        }

        Ok(Module(Rc::new(ModuleInner{
            handle: handle,
            context: self.clone(),
        })))
    }

    /// Constructs a plain device-only buffer
    pub fn buffer(&self) -> BufferBuilder {
        BufferBuilder { context: self.clone() }
    }

    fn poison(&self) {
        self.0.poisoned.set(true);
    }

    fn poisoned(&self) -> bool {
        self.0.poisoned.get()
    }

    /// Sets the maximum heap size useable by kernels in this context.
    /// Device kernels can use `malloc()` and `free()` to access the heap.
    pub fn set_heap_size(&self, size: usize) -> Result<()> {
        unsafe {
            lift(ll::cuCtxSetLimit(
                ll::CUlimit_enum::CU_LIMIT_MALLOC_HEAP_SIZE,
                size,
            ))
        }
    }

    /// Gets the maximum heap size useable by kernels in this context.
    /// Device kernels can use `malloc()` and `free()` to access the heap.
    pub fn get_heap_size(&self) -> Result<usize> {
        let mut size = 0;

        unsafe {
            lift(ll::cuCtxGetLimit(
                &mut size,
                ll::CUlimit_enum::CU_LIMIT_MALLOC_HEAP_SIZE,
            ))?
        }

        Ok(size)
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        if !self.poisoned.get() {
            unsafe { lift(ll::cuCtxDestroy_v2(self.handle)).unwrap() }
        }
    }
}

/// A CUDA device (a GPU)
#[derive(Debug)]
pub struct Device {
    handle: ll::CUdevice,
}

impl Device {
    /// Binds to the `nth` device
    pub fn from_index(nth: u16) -> Result<Self> {
        let mut handle = 0;

        unsafe { lift(ll::cuDeviceGet(&mut handle, i32::from(nth)))? }

        Ok(Device { handle: handle })
    }

    /// Returns the number of available devices
    pub fn count() -> Result<u32> {
        let mut count: i32 = 0;

        unsafe { lift(ll::cuDeviceGetCount(&mut count))? }

        Ok(count as u32)
    }

    /// Creates a CUDA context for this device
    pub fn create_context(&self) -> Result<Context> {
        let mut handle = ptr::null_mut();
        // TODO expose
        let flags = 0;

        unsafe { lift(ll::cuCtxCreate_v2(&mut handle, flags, self.handle))? }

        Ok(Context(Rc::new(ContextInner {
            poisoned: Cell::new(false),
            handle: handle,
        })))
    }

    /// Returns the name of the device
    pub fn name(&self) -> Result<String> {
        let mut s: Vec<u8> = vec![0; 64];
        unsafe {
            lift(ll::cuDeviceGetName(
                s.as_mut_ptr() as *mut i8,
                s.len() as i32,
                self.handle,
            ))?;
        }
        s.retain(|&byte| byte != 0); // throw out nulls
        Ok(CString::new(s).unwrap().into_string().unwrap())
    }

    /// Returns the maximum number of threads a block can have
    pub fn max_threads_per_block(&self) -> Result<i32> {
        use self::ll::CUdevice_attribute_enum::*;

        self.get(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    }

    /// Returns the total amount of (non necessarily free) memory, in bytes,
    /// that the device has
    pub fn total_memory(&self) -> Result<usize> {
        let mut bytes = 0;

        unsafe { lift(ll::cuDeviceTotalMem_v2(&mut bytes, self.handle))? };

        Ok(bytes)
    }

    /// Is the device capable of unified memory (same address between host and device)?
    pub fn has_unified_memory(&self) -> Result<bool> {
        use self::ll::CUdevice_attribute_enum::*;
        Ok(self.get(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)? == 1)
    }

    fn get(&self, attr: ll::CUdevice_attribute) -> Result<i32> {
        let mut value = 0;

        unsafe { lift(ll::cuDeviceGetAttribute(&mut value, attr, self.handle))? }

        Ok(value)
    }
}

/// A function that the CUDA device can execute. AKA a "kernel"
pub struct Function {
    name: String,
    handle: ll::CUfunction,
    module: Module,
}

impl Function {
    /// Execute a function on the GPU
    ///
    /// NOTE This function blocks until the GPU has finished executing the
    /// kernel
    pub fn launch(&self, args: &[&Any], grid: Grid, block: Block) -> Result<()> {
        let stream = Stream::new()?;
        // TODO expose
        let shared_mem_bytes = 0;
        // TODO expose
        let extra = ptr::null_mut();

        unsafe {
            lift(ll::cuLaunchKernel(
                self.handle,
                grid.x,
                grid.y,
                grid.z,
                block.x,
                block.y,
                block.z,
                shared_mem_bytes,
                stream.handle,
                args.as_ptr() as *mut _,
                extra,
            ))?
        }

        let result = stream.sync();
        if result.is_err() {
            // launch error is likely sticky, meaning any future API calls will return same error
            // there's no effective way to ignore it besides destroying the context. However,
            // we should avoid calling destructors on buffers etc since those will error as well,
            // leading to confusing errors while unwinding
            error!("poisoning kernel {}", self.name);
            self.module.0.context.poison();
        }
        stream.destroy()
    }
}

/// A CUDA "grid"
pub struct Grid {
    x: u32,
    y: u32,
    z: u32,
}

impl Grid {
    /// One dimensional grid
    pub fn x(x: u32) -> Self {
        Grid { x: x, y: 1, z: 1 }
    }

    /// Two dimensional grid
    pub fn xy(x: u32, y: u32) -> Self {
        Grid { x: x, y: y, z: 1 }
    }

    /// Three dimensional grid
    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Grid { x: x, y: y, z: z }
    }
}

/// A PTX module
#[derive(Clone)]
pub struct Module(Rc<ModuleInner>);
struct ModuleInner {
    handle: ll::CUmodule,
    context: Context,
}

impl Module {
    /// Retrieves a function from the PTX module
    pub fn function<T: AsRef<str>>(&self, name: T) -> Result<Function> {
        let mut handle = ptr::null_mut();
        let cname = CString::new(name.as_ref()).unwrap();

        unsafe {
            lift(ll::cuModuleGetFunction(
                &mut handle,
                self.0.handle,
                cname.as_ptr(),
            ))?
        }

        Ok(Function {
            name: name.as_ref().into(),
            handle: handle,
            module: self.clone(),
        })
    }

    fn get_symbol_info(&self, symbol: impl AsRef<str>) -> Result<(usize, usize)> {
        let cstr = CString::new(symbol.as_ref()).map_err(|_| Error::InvalidValue)?;
        let mut ptr = 0;
        let mut size = 0;
        unsafe {
            lift(ll::cuModuleGetGlobal_v2(
                &mut ptr as *mut usize as _,
                &mut size as *mut usize as _,
                self.0.handle,
                cstr.as_ptr(),
            ))?;
            Ok((ptr, size))
        }
    }

    /// Return the address of the given symbol
    pub fn get_symbol_address(&self, symbol: impl AsRef<str>) -> Result<usize> {
        Ok(self.get_symbol_info(symbol)?.0)
    }

    /// Returns the size of the given symbol
    pub fn get_symbol_size(&self, symbol: impl AsRef<str>) -> Result<usize> {
        Ok(self.get_symbol_info(symbol)?.1)
    }

    /// Set global symbol
    pub fn set_symbol<T>(&self, symbol: impl AsRef<str>, data: &T) -> Result<()> {
        let (addr, size) = self.get_symbol_info(symbol)?;
        assert!(mem::size_of_val(data) <= size);

        unsafe {
            lift(ll::cuMemcpyHtoD_v2(
                addr as u64,
                data as *const _ as usize as *const _,
                mem::size_of_val(data),
            ))
        }
    }

    /// Reads data stored in global symbol. Chose T correctly
    pub unsafe fn get_symbol<T: Default>(&self, symbol: impl AsRef<str>) -> Result<Box<T>> {
        let (addr, size) = self.get_symbol_info(symbol)?;
        assert!(size >= mem::size_of::<T>());

        let b = Box::new(T::default());
        lift(ll::cuMemcpyDtoH_v2(
            &*b as *const T as *mut _,
            addr as u64,
            mem::size_of::<T>(),
        ))?;
        Ok(b)
    }
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if !self.context.poisoned() {
            unsafe { lift(ll::cuModuleUnload(self.handle)).unwrap() }
        }
    }
}

// TODO expose
struct Stream {
    handle: ll::CUstream,
}

impl Stream {
    fn new() -> Result<Self> {
        let mut handle = ptr::null_mut();
        // TODO expose
        let flags = 0;

        unsafe { lift(ll::cuStreamCreate(&mut handle, flags))? }

        Ok(Stream { handle: handle })
    }

    fn destroy(self) -> Result<()> {
        unsafe { lift(ll::cuStreamDestroy_v2(self.handle)) }
    }

    fn sync(&self) -> Result<()> {
        unsafe { lift(ll::cuStreamSynchronize(self.handle)) }
    }
}

/// Value who's type has been erased
pub enum Any {}

/// Erase the type of a value
#[allow(non_snake_case)]
pub fn Any<T>(ref_: &T) -> &Any {
    unsafe { &*(ref_ as *const T as *const Any) }
}

#[allow(missing_docs)]
#[derive(Debug)]
pub enum Error {
    AlreadyAcquired,
    AlreadyMapped,
    ArrayIsMapped,
    Assert,
    ContextAlreadyCurrent,
    ContextAlreadyInUse,
    ContextIsDestroyed,
    Deinitialized,
    EccUncorrectable,
    FileNotFound,
    HardwareStackError,
    HostMemoryAlreadyRegistered,
    HostMemoryNotRegistered,
    IllegalAddress,
    IllegalInstruction,
    InvalidAddressSpace,
    InvalidContext,
    InvalidDevice,
    InvalidGraphicsContext,
    InvalidHandle,
    InvalidImage,
    InvalidPc,
    InvalidPtx,
    InvalidSource,
    InvalidValue,
    LaunchFailed,
    LaunchIncompatibleTexturing,
    LaunchOutOfResources,
    LaunchTimeout,
    MapFailed,
    MisalignedAddress,
    NoBinaryForGpu,
    NoDevice,
    NotFound,
    NotInitialized,
    NotMapped,
    NotMappedAsArray,
    NotMappedAsPointer,
    NotPermitted,
    NotReady,
    NotSupported,
    NvlinkUncorrectable,
    OperatingSystem,
    OutOfMemory,
    PeerAccessAlreadyEnabled,
    PeerAccessNotEnabled,
    PeerAccessUnsupported,
    PrimaryContextActive,
    ProfilerAlreadyStarted,
    ProfilerAlreadyStopped,
    ProfilerDisabled,
    ProfilerNotInitialized,
    SharedObjectInitFailed,
    SharedObjectSymbolNotFound,
    TooManyPeers,
    Unknown,
    UnmapFailed,
    UnsupportedLimit,
}

impl std::error::Error for Error {}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Wrappers around the memory API
pub mod raw_mem {
    use super::*;

    /// `memcpy` direction
    pub enum Direction {
        /// `src` points to device memory. `dst` points to host memory
        DeviceToHost,
        /// `src` points to host memory. `dst` points to device memory
        HostToDevice,
    }

    // TODO should this be a method of `Context`?
    /// Allocate `n` bytes of memory on the device
    pub unsafe fn allocate(n: usize) -> Result<*mut u8> {
        let mut d_ptr = 0;

        lift(ll::cuMemAlloc_v2(&mut d_ptr, n))?;

        Ok(d_ptr as *mut u8)
    }

    // TODO same question as `allocate`
    /// Free the memory pointed to by `ptr`
    pub unsafe fn deallocate(ptr: *mut u8) -> Result<()> {
        lift(ll::cuMemFree_v2(ptr as u64))
    }

    /// Copy `n` bytes of memory from `src` to `dst`
    ///
    /// `direction` indicates where `src` and `dst` are located (device or host)
    pub unsafe fn copy<T>(
        src: *const T,
        dst: *mut T,
        count: usize,
        direction: Direction,
    ) -> Result<()> {
        use self::Direction::*;

        let bytes = count * mem::size_of::<T>();

        lift(match direction {
            DeviceToHost => ll::cuMemcpyDtoH_v2(dst as *mut _, src as u64, bytes),
            HostToDevice => ll::cuMemcpyHtoD_v2(dst as u64, src as *const _, bytes),
        })?;

        Ok(())
    }
}

/// Represents an unallocated normal buffer
pub struct BufferBuilder {
    context: Context,
}
impl BufferBuilder {
    /// Realize this buffer by allocating enough space for `len` objects
    pub fn alloc<T>(self, len: usize) -> Result<Buffer<T>> {
        unsafe {
            let ptr = raw_mem::allocate(len * mem::size_of::<T>())?;
            Ok(Buffer {
                addr: ptr as usize,
                len: len,
                context: self.context,
                _t: PhantomData,
            })
        }
    }

    /// Construct a buffer from the given iterator
    pub fn from_slice<T, S: AsRef<[T]>>(self, slice: S) -> Result<Buffer<T>> {
        let slice = slice.as_ref();
        let buffer = self.alloc(slice.len())?;
        buffer.copy_from(&slice)?;
        Ok(buffer)
    }

    /// Construct a buffer from the given iterator
    pub fn from_iter<T, I: IntoIterator<Item = T>>(self, i: I) -> Result<Buffer<T>> {
        let data: Vec<T> = i.into_iter().collect();
        let buffer = self.alloc(data.len())?;
        buffer.copy_from(&data)?;
        Ok(buffer)
    }
}

/// Represents a region of device memory which can be allocated using the `Context` struct.
pub struct Buffer<T> {
    addr: usize,
    len: usize,
    context: Context,
    _t: PhantomData<T>,
}
impl<T> Buffer<T> {
    /// Returns number of elements allocated
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns number of bytes allocated
    pub fn size(&self) -> usize {
        self.len * mem::size_of::<T>()
    }

    /// Returns the device memory address, unusable on host
    pub fn addr(&self) -> usize {
        self.addr
    }

    /// Copies memory from the specified host buffer to the device
    pub fn copy_from(&self, data: &[T]) -> Result<()> {
        assert!(data.len() <= self.len());
        unsafe {
            // copy(
            //     data.as_ptr(),
            //     self.0 as _,
            //     data.len() * mem::size_of::<T>(),
            //     Direction::HostToDevice,
            // )
            lift(ll::cuMemcpyHtoD_v2(
                self.addr as _,
                data.as_ptr() as _,
                data.len() * mem::size_of::<T>(),
            ))
        }
    }

    /// Copies memory from device to the specified host buffer
    pub fn copy_to(&self, data: &mut [T]) -> Result<()> {
        unsafe {
            // copy(
            //     self.0 as _,
            //     data.as_mut_ptr(),
            //     self.size().min(data.len() * mem::size_of::<T>()),
            //     Direction::DeviceToHost,
            // )
            lift(ll::cuMemcpyDtoH_v2(
                data.as_mut_ptr() as _,
                self.addr as _,
                self.size().min(data.len() * mem::size_of::<T>()),
            ))
        }
    }

    /// Read data into a `Vec<T>`
    pub fn read_to_vec(&self) -> Result<Vec<T>> {
        let mut v = Vec::with_capacity(self.len());
        unsafe {
            lift(ll::cuMemcpyDtoH_v2(
                v.as_mut_ptr() as _,
                self.addr as _,
                self.size(),
            ))?;
            v.set_len(self.len());
            Ok(v)
        }
    }
}

impl<T: Clone> Buffer<T> {
    /// Initializes the buffer to this value
    ///
    /// Allocates a host-side array of data, then copies
    pub fn fill(&self, data: T) -> Result<()> {
        let v = vec![data; self.len()];
        self.copy_from(&v)
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        if !self.context.poisoned() {
            unsafe {
                raw_mem::deallocate(self.addr as _).expect("Could not free");
            }
        }
    }
}

/// Initialize the CUDA runtime
pub fn initialize() -> Result<()> {
    // TODO expose
    let flags = 0;

    unsafe { lift(ll::cuInit(flags)) }
}

/// Returns the version of the CUDA runtime
pub fn version() -> Result<i32> {
    let mut version = 0;

    unsafe { lift(ll::cuDriverGetVersion(&mut version))? }

    Ok(version)
}

#[allow(missing_docs)]
pub type Result<T> = result::Result<T, Error>;

fn lift(e: ll::CUresult) -> Result<()> {
    use self::ll::cudaError_enum::*;
    use self::Error::*;

    Err(match e {
        CUDA_SUCCESS => return Ok(()),
        CUDA_ERROR_ALREADY_ACQUIRED => AlreadyAcquired,
        CUDA_ERROR_ALREADY_MAPPED => AlreadyMapped,
        CUDA_ERROR_ARRAY_IS_MAPPED => ArrayIsMapped,
        CUDA_ERROR_ASSERT => Assert,
        CUDA_ERROR_CONTEXT_ALREADY_CURRENT => ContextAlreadyCurrent,
        CUDA_ERROR_CONTEXT_ALREADY_IN_USE => ContextAlreadyInUse,
        CUDA_ERROR_CONTEXT_IS_DESTROYED => ContextIsDestroyed,
        CUDA_ERROR_DEINITIALIZED => Deinitialized,
        CUDA_ERROR_ECC_UNCORRECTABLE => EccUncorrectable,
        CUDA_ERROR_FILE_NOT_FOUND => FileNotFound,
        CUDA_ERROR_HARDWARE_STACK_ERROR => HardwareStackError,
        CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => HostMemoryAlreadyRegistered,
        CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => HostMemoryNotRegistered,
        CUDA_ERROR_ILLEGAL_ADDRESS => IllegalAddress,
        CUDA_ERROR_ILLEGAL_INSTRUCTION => IllegalInstruction,
        CUDA_ERROR_INVALID_ADDRESS_SPACE => InvalidAddressSpace,
        CUDA_ERROR_INVALID_CONTEXT => InvalidContext,
        CUDA_ERROR_INVALID_DEVICE => InvalidDevice,
        CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => InvalidGraphicsContext,
        CUDA_ERROR_INVALID_HANDLE => InvalidHandle,
        CUDA_ERROR_INVALID_IMAGE => InvalidImage,
        CUDA_ERROR_INVALID_PC => InvalidPc,
        CUDA_ERROR_INVALID_PTX => InvalidPtx,
        CUDA_ERROR_INVALID_SOURCE => InvalidSource,
        CUDA_ERROR_INVALID_VALUE => InvalidValue,
        CUDA_ERROR_LAUNCH_FAILED => LaunchFailed,
        CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => LaunchIncompatibleTexturing,
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => LaunchOutOfResources,
        CUDA_ERROR_LAUNCH_TIMEOUT => LaunchTimeout,
        CUDA_ERROR_MAP_FAILED => MapFailed,
        CUDA_ERROR_MISALIGNED_ADDRESS => MisalignedAddress,
        CUDA_ERROR_NOT_FOUND => NotFound,
        CUDA_ERROR_NOT_INITIALIZED => NotInitialized,
        CUDA_ERROR_NOT_MAPPED => NotMapped,
        CUDA_ERROR_NOT_MAPPED_AS_ARRAY => NotMappedAsArray,
        CUDA_ERROR_NOT_MAPPED_AS_POINTER => NotMappedAsPointer,
        CUDA_ERROR_NOT_PERMITTED => NotPermitted,
        CUDA_ERROR_NOT_READY => NotReady,
        CUDA_ERROR_NOT_SUPPORTED => NotSupported,
        CUDA_ERROR_NO_BINARY_FOR_GPU => NoBinaryForGpu,
        CUDA_ERROR_NO_DEVICE => NoDevice,
        CUDA_ERROR_OPERATING_SYSTEM => OperatingSystem,
        CUDA_ERROR_OUT_OF_MEMORY => OutOfMemory,
        CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => PeerAccessAlreadyEnabled,
        CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => PeerAccessNotEnabled,
        CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => PeerAccessUnsupported,
        CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => PrimaryContextActive,
        CUDA_ERROR_PROFILER_ALREADY_STARTED => ProfilerAlreadyStarted,
        CUDA_ERROR_PROFILER_ALREADY_STOPPED => ProfilerAlreadyStopped,
        CUDA_ERROR_PROFILER_DISABLED => ProfilerDisabled,
        CUDA_ERROR_PROFILER_NOT_INITIALIZED => ProfilerNotInitialized,
        CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => SharedObjectInitFailed,
        CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => SharedObjectSymbolNotFound,
        CUDA_ERROR_TOO_MANY_PEERS => TooManyPeers,
        CUDA_ERROR_UNKNOWN => Unknown,
        CUDA_ERROR_UNMAP_FAILED => UnmapFailed,
        CUDA_ERROR_UNSUPPORTED_LIMIT => UnsupportedLimit,
        CUDA_ERROR_NVLINK_UNCORRECTABLE => NvlinkUncorrectable,
    })
}
