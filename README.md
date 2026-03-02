# 🔍 HRM Face Recognition Service

> **AI-Powered Facial Recognition Microservice for Employee Attendance Management**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Service Responsibility](#-service-responsibility)
- [Recognition Workflow](#-recognition-workflow)
- [Technology Stack](#-technology-stack)
- [API Communication Flow](#-api-communication-flow)
- [Request/Response Example](#-requestresponse-example)
- [System Integration](#-system-integration)
- [Deployment Overview](#-deployment-overview)
- [Running Locally](#-running-locally)
- [Docker Setup](#-docker-setup)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Project Overview

The **HRM Face Recognition Service** is a specialized Python-based microservice that provides intelligent facial recognition capabilities for employee attendance verification. As a core component of the distributed HRM Attendance Management System, this service handles the computationally intensive task of face matching while operating as an independent, scalable server.

### Key Characteristics

| Aspect | Description |
|--------|-------------|
| **Architecture** | Microservice - decoupled from main backend |
| **Communication** | REST API with Spring Boot backend |
| **Core Function** | Face detection, embedding extraction, and similarity matching |
| **Deployment** | Containerized with Docker |

---

## 🛡️ Service Responsibility

This microservice is designed with a **single responsibility principle**: perform facial recognition and return match results. The main backend retains full control over business logic decisions.

### What This Service Does ✅

- **Face Detection**: Identifies human faces within provided images
- **Feature Extraction**: Generates high-dimensional face embeddings using deep learning models
- **Face Matching**: Compares captured faces against registered employee embeddings
- **CRUD Operations**: Manage employee face embeddings (register, update, delete)
- **Result Reporting**: Returns verification results to the backend

### What This Service Does NOT Do ❌

- ❌ Make attendance decisions (handled by Spring Boot backend)
- ❌ Store employee personal information
- ❌ Manage attendance records
- ❌ Handle authentication/authorization logic

---

## 🔄 Recognition Workflow

The facial recognition process follows a standardized pipeline from image capture to result delivery:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RECOGNITION WORKFLOW                            │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐         ┌──────────────────┐         ┌──────────────┐
  │   Capture    │         │   Spring Boot    │         │   Face Rec   │
  │   Device     │────────▶│     Backend      │────────▶│  Microservice│
  │  (Camera)    │         │                  │         │              │
  └──────────────┘         └──────────────────┘         └──────┬───────┘
                                                               │
                                                               ▼
                                                    ┌──────────────────────┐
                                                    │  1. Decode Base64    │
                                                    │     Image            │
                                                    └──────────┬───────────┘
                                                               ▼
                                                    ┌──────────────────────┐
                                                    │  2. Face Detection   │
                                                    │     (InsightFace)    │
                                                    └──────────┬───────────┘
                                                               ▼
                                                    ┌──────────────────────┐
                                                    │  3. Embedding        │
                                                    │     Extraction       │
                                                    └──────────┬───────────┘
                                                               ▼
                                                    ┌──────────────────────┐
                                                    │  4. Cosine Similarity│
                                                    │     Comparison       │
                                                    └──────────┬───────────┘
                                                               ▼
                                                    ┌──────────────────────┐
                                                    │  5. Threshold-based  │
                                                    │     Match Decision   │
                                                    └──────────┬───────────┘
                                                               │
                                                               ▼
  ┌──────────────┐         ┌──────────────────┐         ┌──────────────┐
  │   Attendance │         │   Spring Boot    │◀────────│   Return     │
  │   Record     │◀────────│     Backend      │         │   Result     │
  │   Updated    │         │  (Decision Maker)│         │              │
  └──────────────┘         └──────────────────┘         └──────────────┘
```

### Workflow Steps

| Step | Process | Description |
|------|---------|-------------|
| 1 | **Image Reception** | Receives Base64-encoded image from backend |
| 2 | **Decoding** | Converts Base64 string to binary image data |
| 3 | **Face Detection** | Uses InsightFace to locate faces in the image |
| 4 | **Embedding Extraction** | Generates 512-dimensional feature vector |
| 5 | **Database Query** | Retrieves all registered embeddings from MySQL |
| 6 | **Similarity Calculation** | Computes cosine similarity against all records |
| 7 | **Match Determination** | Returns best match if similarity ≥ threshold (0.6) |

---

## 🛠️ Technology Stack

### Core Technologies

| Category | Technology | Purpose |
|----------|------------|---------|
| **Framework** | FastAPI | High-performance async web framework |
| **Language** | Python 3.11 | Primary development language |
| **ML/AI** | InsightFace | State-of-the-art face analysis library |
| **Runtime** | ONNX Runtime | Optimized deep learning inference |
| **Computer Vision** | OpenCV | Image processing and manipulation |

### Database & Storage

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Database** | MySQL | Persistent storage for face embeddings |
| **ORM** | SQLAlchemy 2.0 | Database abstraction layer |
| **Driver** | PyMySQL | MySQL database connectivity |

### Infrastructure

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Server** | Uvicorn | ASGI server for production |
| **Container** | Docker | Application containerization |
| **Base Image** | Python 3.11-slim | Minimal Python runtime |

### Python Dependencies

```
fastapi                    # Web framework
uvicorn[standard]          # ASGI server
numpy                      # Numerical computations
opencv-python-headless     # Computer vision
pillow                     # Image processing
python-multipart           # File upload support
insightface                # Face recognition engine
onnxruntime                # ML model inference
SQLAlchemy==2.0.44         # ORM
PyMySQL==1.1.2             # MySQL driver
cryptography               # Security utilities
```

---

## 🌐 API Communication Flow

The microservice exposes RESTful endpoints under the `/facial-recognition` prefix.

### Base URL

```
http://<service-host>:8000/facial-recognition
```

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/hello` | Health check endpoint |
| `POST` | `/register-face` | Register single employee with face images |
| `POST` | `/register-face-batch` | Bulk register via ZIP upload |
| `POST` | `/face-recognition` | Perform face recognition |
| `PUT` | `/update-face` | Update employee face embeddings |
| `DELETE` | `/delete-face` | Remove employee face data |

### Communication Sequence

```
Spring Boot Backend                          Face Recognition Service
        │                                              │
        │───────── POST /face-recognition ────────────▶│
        │         { "image": "base64_encoded" }        │
        │                                              │
        │                                              │──▶ Process image
        │                                              │──▶ Extract embedding
        │                                              │──▶ Compare with DB
        │                                              │
        │◀─────────── JSON Response ───────────────────│
        │         { "employee_id": "EMP001" }          │
        │                   or                         │
        │         { "message": "No match found" }      │
        │                                              │
```

---

## 📡 Request/Response Example

### Face Recognition Request

**Endpoint:** `POST /facial-recognition/face-recognition`

**Headers:**
```http
Content-Type: application/json
Accept: application/json
```

**Request Body:**
```json
{
  "image": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

### Success Response (Match Found)

**Status:** `200 OK`

```json
{
  "employee_id": "EMP001"
}
```

### Success Response (No Match)

**Status:** `200 OK`

```json
{
  "message": "No match found"
}
```

### Error Response

**Status:** `500 Internal Server Error`

```json
{
  "detail": "Failed to process image"
}
```

### Face Registration Request

**Endpoint:** `POST /facial-recognition/register-face`

**Request Body:**
```json
{
  "employee_id": "EMP001",
  "images": [
    "/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "/9j/4AAQSkZJRgABAQAAAQABAAD..."
  ]
}
```

**Response:**
```json
{
  "message": "Registered successfully",
  "embeddings": 2
}
```

---

## 🔗 System Integration

### Architecture Context

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HRM ATTENDANCE MANAGEMENT SYSTEM                     │
└─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐
  │   Frontend App  │
  │  (Web/Mobile)   │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    SPRING BOOT BACKEND                              │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
  │  │   Attendance │  │   Employee   │  │   Report     │              │
  │  │   Service    │  │   Service    │  │   Service    │              │
  │  └──────────────┘  └──────────────┘  └──────────────┘              │
  │                                                                     │
  │  ┌─────────────────────────────────────────────────────────────┐   │
  │  │              Attendance Decision Engine                      │   │
  │  │  • Validate recognition result                               │   │
  │  │  • Check attendance policies                                 │   │
  │  │  • Record attendance status (Present/Absent/Late)            │   │
  │  └─────────────────────────────────────────────────────────────┘   │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 │ REST API
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │              FACE RECOGNITION SERVICE (This Service)                │
  │                                                                     │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
  │  │   Face       │  │   Embedding  │  │   Similarity │              │
  │  │   Detection  │  │   Extraction │  │   Matching   │              │
  │  └──────────────┘  └──────────────┘  └──────────────┘              │
  │                                                                     │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 │ SQLAlchemy ORM
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         MYSQL DATABASE                              │
  │                    (Face Embeddings Storage)                        │
  └─────────────────────────────────────────────────────────────────────┘
```

### Integration Contract

| Aspect | Specification |
|--------|---------------|
| **Protocol** | HTTP/HTTPS |
| **Data Format** | JSON |
| **Image Encoding** | Base64 strings |
| **Response Time** | Target < 2 seconds per recognition |
| **Authentication** | Handled by backend (service-to-service) |

---

## 🚀 Deployment Overview

### Container Architecture

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY .env ./
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Deployment Checklist

- [ ] Environment variables configured (`.env` file)
- [ ] MySQL database accessible
- [ ] Port 8000 exposed
- [ ] Container resource limits set
- [ ] Logging configured
- [ ] Health checks implemented

---

## 💻 Running Locally

### Prerequisites

- Python 3.11+
- MySQL 8.0+
- Virtual environment tool (venv/conda)

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/TRONGG2005k/facial_recognition_api.git
cd facial_recognition_api

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment variables
copy .env.example .env
# Edit .env with your database credentials

# 6. Run the application
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Verify Installation

```bash
# Health check
curl http://localhost:8000/facial-recognition/hello

# Expected response
{"message": "Hello world"}
```

### API Documentation (Swagger UI)

Once running, access interactive API docs at:

```
http://localhost:8000/docs
```

---

## 🐳 Docker Setup

### Build and Run

```bash
# Build the Docker image
docker build -t hrm-face-recognition .

# Run the container
docker run -d \
  --name face-recognition-service \
  -p 8000:8000 \
  --env-file .env \
  hrm-face-recognition
```

### Docker Compose (Recommended)

```yaml
version: '3.8'

services:
  face-recognition:
    build: .
    container_name: hrm-face-recognition
    ports:
      - "8000:8000"
    env_file:
      - .env
    networks:
      - hrm-network
    restart: unless-stopped

networks:
  hrm-network:
    driver: bridge
```

### Container Management

```bash
# View logs
docker logs -f hrm-face-recognition

# Stop container
docker stop hrm-face-recognition

# Remove container
docker rm hrm-face-recognition
```

---

## 🔮 Future Improvements

Planned enhancements for upcoming releases:

| Priority | Enhancement | Description |
|----------|-------------|-------------|
| 🔴 High | **Async Processing** | Implement background task queue for batch operations |
| 🔴 High | **Model Optimization** | Quantize models for faster inference |
| 🟡 Medium | **Multi-face Detection** | Support recognition of multiple faces in single image |
| 🟡 Medium | **Confidence Scoring** | Return similarity scores for matches |
| 🟡 Medium | **API Rate Limiting** | Implement request throttling |
| 🟢 Low | **Metrics & Monitoring** | Add Prometheus metrics export |
| 🟢 Low | **Health Check Endpoint** | Dedicated `/health` for load balancers |

---

## 👤 Author

**HRM Development Team**

- 🔗 **Repository**: [facial_recognition_api](https://github.com/TRONGG2005k/facial_recognition_api)
- 📧 **Contact**: Contact the HRM team for support

### Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass before submitting PR
- Documentation is updated for new features

---

## 📄 License

This project is proprietary software developed for the HRM Attendance Management System.

---

<div align="center">

**Made with ❤️ for smarter attendance management**

</div>
