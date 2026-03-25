"""
Service xử lý batch registration từ file ZIP.
"""

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from sqlalchemy.orm import Session

from src.dto.response.UploadZipResponse import UploadZipResponse
from src.service.FaceRecognitionService import FaceRecognitionService

logger = logging.getLogger(__name__)

# Constants
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
FACE_FOLDER = "FACE"
MIN_PATH_PARTS = 3  # employee_id/FACE/image.jpg


@dataclass
class ZipEntryInfo:
    """Thông tin về một entry trong ZIP file."""
    employee_id: str
    filename: str
    image_data: bytes


@dataclass
class BatchResult:
    """Kết quả xử lý batch."""
    total_embeddings: int = 0
    total_employees: int = 0
    total_success: int = 0
    total_failed: int = 0
    error_messages: list[str] | None = None
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []


class FaceBatchService:
    """
    Service xử lý đăng ký khuôn mặt hàng loạt từ file ZIP.
    
    Cấu trúc ZIP file mong đợi:
        employee_id/FACE/image.jpg
    """

    def __init__(self, face_service: FaceRecognitionService):
        self.face_service = face_service

    def register_from_zip(
        self,
        db: Session,
        zip_file: BinaryIO
    ) -> UploadZipResponse:
        """
        Đăng ký khuôn mặt hàng loạt từ file ZIP.

        Args:
            db: Database session
            zip_file: File ZIP chứa ảnh khuôn mặt

        Returns:
            UploadZipResponse: Kết quả xử lý
        """
        logger.debug("Starting batch registration from ZIP file")

        result = BatchResult()
        entries = []
        current_employee: str | None = None
        employee_images: list[bytes] = []

        try:
            with zipfile.ZipFile(zip_file, "r") as z:
                logger.debug(f"ZIP file contains {len(z.infolist())} entries")
                
                for entry in z.infolist():
                    if entry.is_dir():
                        continue

                    zip_info = self._process_zip_entry(z, entry, result)
                    if zip_info is None:
                        continue

                    # Nếu đổi employee, lưu employee cũ trước
                    if current_employee and zip_info.employee_id != current_employee:
                        self._save_employee_embeddings(
                            db, current_employee, employee_images, result
                        )
                        employee_images.clear()

                    current_employee = zip_info.employee_id
                    employee_images.append(zip_info.image_data)

                # Lưu employee cuối cùng
                if current_employee and employee_images:
                    self._save_employee_embeddings(
                        db, current_employee, employee_images, result
                    )

        except zipfile.BadZipFile as e:
            logger.exception("Invalid ZIP file")
            result.error_messages.append(f"Lỗi ZIP: {str(e)}")
            result.total_failed += 1
        except Exception as e:
            logger.exception("Error processing ZIP file")
            result.error_messages.append(f"Lỗi: {str(e)}")
            result.total_failed += 1

        return self._build_response(result)

    def _process_zip_entry(
        self,
        zip_ref: zipfile.ZipFile,
        entry: zipfile.ZipInfo,
        result: BatchResult
    ) -> ZipEntryInfo | None:
        """
        Xử lý một entry trong ZIP file.

        Args:
            zip_ref: Reference đến ZIP file
            entry: ZipInfo entry
            result: Object lưu kết quả

        Returns:
            ZipEntryInfo nếu hợp lệ, None nếu lỗi
        """
        result.total_success += 1  # Count total entries processed
        filename = entry.filename

        # Kiểm tra định dạng file
        if not self._is_valid_image(filename):
            result.error_messages.append(f"{filename}: không phải file ảnh hợp lệ")
            result.total_failed += 1
            return None

        # Parse path
        path_parts = Path(filename).parts
        if len(path_parts) < MIN_PATH_PARTS:
            result.error_messages.append(f"{filename}: sai cấu trúc thư mục")
            result.total_failed += 1
            return None

        employee_id = path_parts[0]
        folder = path_parts[1]

        if folder != FACE_FOLDER:
            result.error_messages.append(f"{filename}: không nằm trong folder {FACE_FOLDER}")
            result.total_failed += 1
            return None

        # Đọc dữ liệu ảnh
        try:
            with zip_ref.open(entry) as f:
                image_data = f.read()
                return ZipEntryInfo(
                    employee_id=employee_id,
                    filename=filename,
                    image_data=image_data
                )
        except Exception as e:
            logger.exception(f"Error reading file {filename}")
            result.error_messages.append(f"{filename}: {str(e)}")
            result.total_failed += 1
            return None

    def _is_valid_image(self, filename: str) -> bool:
        """Kiểm tra file có phải định dạng ảnh hợp lệ."""
        return filename.lower().endswith(VALID_IMAGE_EXTENSIONS)

    def _save_employee_embeddings(
        self,
        db: Session,
        employee_id: str,
        images: list[bytes],
        result: BatchResult
    ) -> None:
        """
        Lưu embeddings cho một nhân viên.

        Args:
            db: Database session
            employee_id: ID nhân viên
            images: Danh sách ảnh
            result: Object lưu kết quả
        """
        if not images:
            return

        try:
            count = self.face_service.register_embeddings(db, employee_id, images)

            if count > 0:
                result.total_embeddings += count
                result.total_employees += 1
                result.total_success += len(images)
            else:
                result.total_failed += len(images)
                result.error_messages.append(
                    f"{employee_id}: không tạo được embedding"
                )
        except Exception as e:
            logger.exception(f"Error saving embeddings for {employee_id}")
            result.total_failed += len(images)
            result.error_messages.append(f"{employee_id}: {str(e)}")

    def _build_response(self, result: BatchResult) -> UploadZipResponse:
        """Build response từ kết quả xử lý."""
        return UploadZipResponse(
            error_message=result.error_messages,
            total=result.total_success + result.total_failed,
            failed_total=result.total_failed,
            total_success=result.total_success
        )
