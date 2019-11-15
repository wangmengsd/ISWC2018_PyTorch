from pynvml import *

class PrintGPUStatus:
    def __init__(self):
        print "this class is for printing GPU status"

    @staticmethod
    def print_gpu_status(note):
        print note
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            print "GPU: " + str(i) + ", Used: " + str(info.used / (1024 ** 2)) + "MB, Free: " + str(info.free / (1024 ** 2)) + "MB, Total: " + str(info.total / (1024 ** 2)) + "MB"
        nvmlShutdown()


if __name__ == "__main__":
    PrintGPUStatus.print_gpu_status("test")