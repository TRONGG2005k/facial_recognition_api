class UploadZipResponse:
    def __init__(self, errorMessage: list[str], total: int, failedTotal: int, totalSuccess:int):
        self.errorMessage = errorMessage
        self.total = total
        self.failedTotal = failedTotal
        self.totalSuccess = totalSuccess