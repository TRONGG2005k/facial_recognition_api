import logging
import zipfile
from typing import BinaryIO
from sqlalchemy.orm import Session
from src.dto.response.zip_up_load_response import UploadZipResponse
from src.service.FaceRecognitionService import FaceRecognitionService

logger = logging.getLogger(__name__)


class FaceBatchService:

    def __init__(self, face_service: FaceRecognitionService):
        self.face_service = face_service

    def register_from_zip(
        self,
        db: Session,
        zip_file: BinaryIO
    ) -> UploadZipResponse:

        logger.debug("Starting batch registration from ZIP file")

        total_embeddings = 0
        total_employees = 0

        total_success = 0;
        total_failed = 0;
        error_message = [];
        total = 0;
        try:
            with zipfile.ZipFile(zip_file, "r") as z:

                logger.debug(f"ZIP file contains {len(z.infolist())} entries")

                current_employee = None
                images: list[bytes] = []

                for entry in z.infolist():

                    if entry.is_dir():
                        continue

                    try:
                        total += 1

                        filename = entry.filename

                        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                            error_message.append(f"{filename} không phải file ảnh")
                            total_failed += 1
                            continue

                        parts = filename.split("/")

                        if len(parts) < 3:
                            error_message.append(f"{filename} sai cấu trúc thư mục")
                            total_failed += 1
                            continue

                        employee_id = parts[0]
                        folder = parts[1]

                        if folder != "FACE":
                            error_message.append(f"{filename} không nằm trong folder FACE")
                            total_failed += 1
                            continue

                        if current_employee and employee_id != current_employee:

                            count = self.face_service.register_embeddings(
                                db,
                                current_employee,
                                images
                            )

                            if count > 0:
                                total_embeddings += count
                                total_employees += 1
                                total_success += len(images)
                            else:
                                total_failed += len(images)
                                error_message.append(
                                    f"{current_employee}: không tạo được embedding"
                                )

                            images.clear()

                        current_employee = employee_id

                        with z.open(entry) as f:
                            images.append(f.read())

                    except Exception as e:
                        logger.exception(f"Lỗi khi xử lý file {entry.filename}")
                        error_message.append(f"{entry.filename}: {str(e)}")
                        total_failed += 1

                # last employee
                if current_employee and images:
                    try:
                        count = self.face_service.register_embeddings(
                            db,
                            current_employee,
                            images
                        )

                        if count > 0:
                            total_embeddings += count
                            total_employees += 1
                            total_success += len(images)
                        else:
                            total_failed += len(images)
                            error_message.append(
                                f"{current_employee}: không tạo được embedding"
                            )

                    except Exception as e:
                        logger.exception(
                            f"Lỗi khi lưu embeddings cho {current_employee}"
                        )
                        error_message.append(
                            f"{current_employee}: {str(e)}"
                        )
                        total_failed += len(images)

        except Exception as e:
            logger.exception("Lỗi khi xử lý ZIP file")
            error_message.append(f"Lỗi ZIP: {str(e)}")
            total_failed+=1

            
        return  UploadZipResponse(
            errorMessage=error_message,
            total=total,
            totalSuccess=total_success,
            failedTotal=total_failed
        )
