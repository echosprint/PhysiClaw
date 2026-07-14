"""UVC exposure control over IOKit — pure ctypes. The IOKit machinery is
macOS-only; `ticks_from_log2_seconds` (the UVC tick scale) is shared with
the Linux backend.

macOS offers no API OpenCV can reach to control a UVC webcam's exposure:
the AVFoundation backend never implemented the exposure properties, and
Apple's manual-exposure API doesn't apply to external UVC cameras. The
camera itself, however, honors standard UVC control requests over USB.
This module sends them through IOKit's USB plug-in interfaces — the one
channel that coexists with Apple's attached driver (libusb-based stacks
must *claim* the interface away from it, which Darwin forbids without
root; the IOKit route needs no privileges and works mid-stream).

Everything here is ~the essential 5-call path of jtfrey/uvc-util, ported
to ctypes so nothing needs a compiler:

    registry walk -> device plug-in -> CreateInterfaceIterator(VideoControl)
    -> interface plug-in -> ControlRequest

ABI notes, because ctypes gets no compiler to check them:
- The vtable layouts mirror IOUSBDeviceStruct100 / IOUSBInterfaceStruct100
  in IOUSBLib.h verbatim (COM: _reserved + IUnknown triple, then slots in
  header order). Only the slots we call get real signatures; the rest are
  opaque pointers, so a miscount would misdirect calls — don't reorder.
- The UUIDs are COM interface constants, verbatim from IOCFPlugIn.h /
  IOUSBLib.h; they are frozen ABI (that is their entire purpose).
- ControlRequest on pipe 0 works even when USBInterfaceOpen is refused
  with kIOReturnExclusiveAccess (Apple's driver holds the interface); the
  open is attempted best-effort and its failure ignored, matching what we
  measured on real hardware.

Fail-soft contract: `exposure_channel()` returns None unless exactly one
VideoControl-capable device is on the bus AND it answers a GET_CUR for
its exposure controls. Callers treat None as "not tunable" and leave the
firmware auto-exposure alone.
"""

import ctypes as C
import logging

log = logging.getLogger(__name__)

# COM interface UUIDs from IOCFPlugIn.h / IOUSBLib.h (frozen ABI).
_PLUGIN_IID = bytes.fromhex("c244e858109c11d491d40050e4c6426f")
_DEV_UC = bytes.fromhex("9dc7b7809ec011d4a54f000a27052861")
_DEV_IID = bytes.fromhex("5c8187d09ef311d48b45000a27052861")
_INTF_UC = bytes.fromhex("2d9786c69ef311d4ad51000a27052861")
_INTF_IID = bytes.fromhex("73c97ae89ef311d4b1d0000a27052861")

# USB / UVC protocol constants (USB Video Class spec 1.1 §4).
_USB_CLASS_VIDEO, _SUBCLASS_VIDEOCONTROL = 14, 1
_DONT_CARE = 0xFFFF
_SET_CUR, _GET_CUR, _GET_MIN, _GET_MAX = 0x01, 0x81, 0x82, 0x83
_CT_AE_MODE, _CT_EXPOSURE_ABS = 0x02, 0x04
# bmRequestType: direction | class request | interface recipient.
_OUT, _IN = 0x21, 0xA1
# Camera-terminal entity ids to try, in order. The UVC spec only requires
# ids to be unique — 1 is the near-universal convention (Linux gadget
# defaults, most firmware), but `probe` verifies rather than assumes: the
# exposure controls must answer with their exact camera-terminal byte
# sizes (AE mode 1, exposure-time 4). The size check is what makes
# probing ids beyond 1 safe — a Processing Unit reached by mistake maps
# these selectors to brightness/contrast (2 bytes each) and is rejected
# by the wLenDone mismatch instead of being silently mis-driven.
_CAMERA_TERMINAL_CANDIDATES = (1, 2, 3, 4)


def ticks_from_log2_seconds(exposure: int) -> int:
    """exposure.py's shared log2-seconds scale (-11 ≈ 0.5ms … -2 = 250ms)
    → EXPOSURE_TIME_ABSOLUTE 100µs ticks, floored at 1. One log2 stop
    stays one halving of ticks, keeping `converge`'s integer stepping a
    real one-stop move. Used by darwin (UVC over IOKit) and linux —
    V4L2's exposure control on USB cameras is defined in the same UVC
    ticks, so the two platforms must never diverge on this scale."""
    return max(1, round((2.0**exposure) * 10_000))


_iokit = None
_cf = None


class _CFUUIDBytes(C.Structure):
    # 16 scalar fields, deliberately not `c_uint8 * 16`: ctypes mishandles
    # by-value structs that contain array fields on Darwin/arm64
    # (python/cpython#110190); flattened scalars take the safe path.
    _fields_ = [(f"b{i}", C.c_uint8) for i in range(16)]


class _FindInterfaceRequest(C.Structure):
    _fields_ = [
        ("bInterfaceClass", C.c_uint16),
        ("bInterfaceSubClass", C.c_uint16),
        ("bInterfaceProtocol", C.c_uint16),
        ("bAlternateSetting", C.c_uint16),
    ]


class _DevRequest(C.Structure):
    _fields_ = [
        ("bmRequestType", C.c_uint8),
        ("bRequest", C.c_uint8),
        ("wValue", C.c_uint16),
        ("wIndex", C.c_uint16),
        ("wLength", C.c_uint16),
        ("pData", C.c_void_p),
        ("wLenDone", C.c_uint32),
    ]


_KR = C.c_int32
_QI = C.CFUNCTYPE(C.c_int32, C.c_void_p, _CFUUIDBytes, C.POINTER(C.c_void_p))
_RELEASE = C.CFUNCTYPE(C.c_uint32, C.c_void_p)
_GET_U16 = C.CFUNCTYPE(_KR, C.c_void_p, C.POINTER(C.c_uint16))
_GET_U8 = C.CFUNCTYPE(_KR, C.c_void_p, C.POINTER(C.c_uint8))
_NOARG = C.CFUNCTYPE(_KR, C.c_void_p)
_ITER = C.CFUNCTYPE(
    _KR, C.c_void_p, C.POINTER(_FindInterfaceRequest), C.POINTER(C.c_uint32)
)
_CTRL = C.CFUNCTYPE(_KR, C.c_void_p, C.c_uint8, C.POINTER(_DevRequest))
_PAD = C.c_void_p  # vtable slot we never call


def _vtbl(*slots):
    """COM object layout: _reserved + IUnknown triple + interface slots."""
    return [
        ("_reserved", _PAD),
        ("QueryInterface", _QI),
        ("AddRef", _PAD),
        ("Release", _RELEASE),
        *slots,
    ]


class _DeviceVtbl(C.Structure):
    # IOUSBDeviceStruct100, slots in IOUSBLib.h order.
    _fields_ = _vtbl(
        *[
            (n, _PAD)
            for n in (
                "CreateDeviceAsyncEventSource",
                "GetDeviceAsyncEventSource",
                "CreateDeviceAsyncPort",
                "GetDeviceAsyncPort",
                "USBDeviceOpen",
                "USBDeviceClose",
                "GetDeviceClass",
                "GetDeviceSubClass",
                "GetDeviceProtocol",
                "GetDeviceVendor",
                "GetDeviceProduct",
                "GetDeviceReleaseNumber",
                "GetDeviceAddress",
                "GetDeviceBusPowerAvailable",
                "GetDeviceSpeed",
                "GetNumberOfConfigurations",
                "GetLocationID",
                "GetConfigurationDescriptorPtr",
                "GetConfiguration",
                "SetConfiguration",
                "GetBusFrameNumber",
                "ResetDevice",
                "DeviceRequest",
                "DeviceRequestAsync",
            )
        ],
        ("CreateInterfaceIterator", _ITER),
    )


class _IntfVtbl(C.Structure):
    # IOUSBInterfaceStruct100, slots in IOUSBLib.h order.
    _fields_ = _vtbl(
        *[
            (n, _PAD)
            for n in (
                "CreateInterfaceAsyncEventSource",
                "GetInterfaceAsyncEventSource",
                "CreateInterfaceAsyncPort",
                "GetInterfaceAsyncPort",
            )
        ],
        ("USBInterfaceOpen", _NOARG),
        ("USBInterfaceClose", _NOARG),
        *[
            (n, _PAD)
            for n in (
                "GetInterfaceClass",
                "GetInterfaceSubClass",
                "GetInterfaceProtocol",
                "GetDeviceVendor",
                "GetDeviceProduct",
                "GetDeviceReleaseNumber",
                "GetConfigurationValue",
            )
        ],
        ("GetInterfaceNumber", _GET_U8),
        *[
            (n, _PAD)
            for n in (
                "GetAlternateSetting",
                "GetNumEndpoints",
                "GetLocationID",
                "GetDevice",
                "SetAlternateInterface",
                "GetBusFrameNumber",
            )
        ],
        ("ControlRequest", _CTRL),
    )


def _load() -> bool:
    global _iokit, _cf
    if _iokit is not None:
        return True
    try:
        _iokit = C.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
        _cf = C.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
    except OSError as exc:  # not macOS, or a very stripped system
        log.info("IOKit unavailable: %s", exc)
        return False
    _cf.CFUUIDGetConstantUUIDWithBytes.restype = C.c_void_p
    _cf.CFUUIDGetConstantUUIDWithBytes.argtypes = [C.c_void_p] + [C.c_uint8] * 16
    _iokit.IOServiceMatching.restype = C.c_void_p
    _iokit.IOServiceMatching.argtypes = [C.c_char_p]
    _iokit.IOServiceGetMatchingServices.argtypes = [
        C.c_uint32,
        C.c_void_p,
        C.POINTER(C.c_uint32),
    ]
    _iokit.IOIteratorNext.restype = C.c_uint32
    _iokit.IOIteratorNext.argtypes = [C.c_uint32]
    _iokit.IOObjectRelease.argtypes = [C.c_uint32]
    _iokit.IOCreatePlugInInterfaceForService.argtypes = [
        C.c_uint32,
        C.c_void_p,
        C.c_void_p,
        C.POINTER(C.POINTER(C.POINTER(C.c_void_p))),
        C.POINTER(C.c_int32),
    ]
    _iokit.IODestroyPlugInInterface.argtypes = [C.POINTER(C.POINTER(C.c_void_p))]
    return True


def _cfuuid(raw: bytes):
    return _cf.CFUUIDGetConstantUUIDWithBytes(None, *raw)


def _uuid_struct(raw: bytes) -> _CFUUIDBytes:
    return _CFUUIDBytes(*raw)


def _call(ptr, name, *args):
    return getattr(ptr.contents.contents, name)(C.cast(ptr, C.c_void_p), *args)


def _com_interface(service: int, uc_uuid: bytes, iid: bytes, vtbl_type):
    """service -> intermediate plug-in -> QueryInterface -> typed COM ptr."""
    plug = C.POINTER(C.POINTER(C.c_void_p))()
    score = C.c_int32()
    kr = _iokit.IOCreatePlugInInterfaceForService(
        service, _cfuuid(uc_uuid), _cfuuid(_PLUGIN_IID), C.byref(plug), C.byref(score)
    )
    if kr != 0 or not plug:
        return None
    # Every COM object shares the IUnknown header, so any vtbl works for QI.
    obj = C.cast(plug, C.POINTER(C.POINTER(_DeviceVtbl)))
    out = C.c_void_p()
    hr = obj.contents.contents.QueryInterface(
        C.cast(obj, C.c_void_p), _uuid_struct(iid), C.byref(out)
    )
    _iokit.IODestroyPlugInInterface(plug)
    if hr != 0 or not out.value:
        return None
    return C.cast(out.value, C.POINTER(C.POINTER(vtbl_type)))


def _video_control_interfaces() -> list:
    """One COM interface pointer per USB device that has a VideoControl
    interface. IOUSBHostDevice is the registry class on 10.15+; the
    IOUSBDevice fallback covers older systems (never both — on modern
    macOS they alias the same devices and would double-count)."""
    for cls in (b"IOUSBHostDevice", b"IOUSBDevice"):
        found = []
        it = C.c_uint32()
        if (
            _iokit.IOServiceGetMatchingServices(
                0, _iokit.IOServiceMatching(cls), C.byref(it)
            )
            != 0
        ):
            continue
        while svc := _iokit.IOIteratorNext(it.value):
            dev = _com_interface(svc, _DEV_UC, _DEV_IID, _DeviceVtbl)
            _iokit.IOObjectRelease(svc)
            if dev is None:
                continue
            req = _FindInterfaceRequest(
                _USB_CLASS_VIDEO, _SUBCLASS_VIDEOCONTROL, _DONT_CARE, _DONT_CARE
            )
            i_it = C.c_uint32()
            kr = _call(dev, "CreateInterfaceIterator", C.byref(req), C.byref(i_it))
            vc_svc = _iokit.IOIteratorNext(i_it.value) if kr == 0 else 0
            if kr == 0:
                _iokit.IOObjectRelease(i_it.value)
            _call(dev, "Release")
            if not vc_svc:
                continue
            intf = _com_interface(vc_svc, _INTF_UC, _INTF_IID, _IntfVtbl)
            _iokit.IOObjectRelease(vc_svc)
            if intf is not None:
                found.append(intf)
        _iokit.IOObjectRelease(it.value)
        if found:
            return found
    return []


class ExposureChannel:
    """Exposure controls of one UVC camera. Obtain via `exposure_channel`."""

    def __init__(self, intf) -> None:
        self._intf = intf
        self._terminal = _CAMERA_TERMINAL_CANDIDATES[0]
        self._range: tuple[int | None, int | None] | None = None
        ifnum = C.c_uint8()
        _call(intf, "GetInterfaceNumber", C.byref(ifnum))
        self._ifnum = ifnum.value
        # Best-effort: instantiates the control pipe on some stacks. A
        # kIOReturnExclusiveAccess refusal (Apple's driver holds the
        # interface) is expected and harmless — pipe-0 control requests
        # go through regardless (measured on real hardware).
        _call(intf, "USBInterfaceOpen")

    def _request(
        self, direction: int, request: int, selector: int, data: bytearray
    ) -> bool:
        """One control request; True only on success AND a full transfer.

        The wLenDone check is load-bearing twice over: it rejects short
        reads, and during terminal discovery it is what distinguishes a
        real camera terminal from some other entity whose same-numbered
        selector has a different byte size."""
        buf = (C.c_uint8 * len(data)).from_buffer(data)
        req = _DevRequest(
            direction,
            request,
            selector << 8,
            (self._terminal << 8) | self._ifnum,
            len(data),
            C.cast(buf, C.c_void_p),
            0,
        )
        kr = _call(self._intf, "ControlRequest", 0, C.byref(req))
        return kr == 0 and req.wLenDone == len(data)

    def _get(self, request: int, selector: int, length: int) -> int | None:
        data = bytearray(length)
        if not self._request(_IN, request, selector, data):
            return None
        return int.from_bytes(data, "little")

    def _set(self, selector: int, value: int, length: int) -> bool:
        return self._request(
            _OUT, _SET_CUR, selector, bytearray(value.to_bytes(length, "little"))
        )

    def _answers_exposure(self) -> bool:
        """Both exposure controls answer at the current terminal id, with
        their exact camera-terminal byte sizes."""
        return (
            self._get(_GET_CUR, _CT_AE_MODE, 1) is not None
            and self._get(_GET_CUR, _CT_EXPOSURE_ABS, 4) is not None
        )

    def probe(self) -> bool:
        """Find the camera terminal. Almost always entity 1; walk the
        candidates for the rare firmware that numbers it differently.
        Requests to a wrong id are STALLed by the device (harmless)."""
        for terminal in _CAMERA_TERMINAL_CANDIDATES:
            self._terminal = terminal
            if self._answers_exposure():
                if terminal != _CAMERA_TERMINAL_CANDIDATES[0]:
                    log.info("UVC camera terminal at entity id %d", terminal)
                return True
        return False

    def set_auto(self) -> bool:
        """Firmware auto-exposure: mode 8 (aperture priority, the usual
        UVC "auto"), falling back to 2 (full auto) where unsupported."""
        return self._set(_CT_AE_MODE, 8, 1) or self._set(_CT_AE_MODE, 2, 1)

    def set_manual(self, ticks: int) -> bool:
        """Manual mode + absolute exposure time in 100µs ticks, clamped
        to the device-reported range when the device publishes one.
        The range is device-static, so it is read once and cached —
        `converge` calls this per step and shouldn't pay two extra
        round-trips each time."""
        if not self._set(_CT_AE_MODE, 1, 1):
            return False
        if self._range is None:
            self._range = (
                self._get(_GET_MIN, _CT_EXPOSURE_ABS, 4),
                self._get(_GET_MAX, _CT_EXPOSURE_ABS, 4),
            )
        lo, hi = self._range
        if lo is not None and hi is not None and lo <= hi:
            ticks = max(lo, min(hi, ticks))
        return self._set(_CT_EXPOSURE_ABS, max(1, ticks), 4)


def exposure_channel() -> ExposureChannel | None:
    """The single UVC camera's exposure channel, or None.

    None when IOKit is unavailable, no UVC device exists, several exist
    (OpenCV indexes cameras via AVFoundation, this module via IOKit, and
    nothing guarantees the two orders agree — with one device there is
    nothing to disagree about), or the device won't answer for its
    exposure controls. Callers treat None as "not tunable"."""
    if not _load():
        return None
    interfaces = _video_control_interfaces()
    if len(interfaces) != 1:
        if len(interfaces) > 1:
            log.info(
                "UVC exposure tuning off: %d UVC devices on the bus "
                "(need exactly 1 to match OpenCV's camera unambiguously)",
                len(interfaces),
            )
        for intf in interfaces:
            _call(intf, "Release")
        return None
    channel = ExposureChannel(interfaces[0])
    if not channel.probe():
        log.info("UVC exposure tuning off: device doesn't answer for exposure")
        _call(interfaces[0], "Release")
        return None
    return channel
