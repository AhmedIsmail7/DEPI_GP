# test_ingest.py
"""
Unit tests for SecureVideoIngestion class
Author: Ramy Safwat
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from modules.ingest import SecureVideoIngestion


@pytest.fixture
def temp_dirs():
    """Fixture to create temporary directories for testing"""
    with tempfile.TemporaryDirectory() as downloads_dir:
        with tempfile.TemporaryDirectory() as metadata_dir:
            yield Path(downloads_dir), Path(metadata_dir)


@pytest.fixture
def ingestion(temp_dirs):
    """Fixture to initialize SecureVideoIngestion with temp dirs"""
    downloads, metadata = temp_dirs
    return SecureVideoIngestion(
        output_dir=str(downloads),
        metadata_dir=str(metadata),
        enable_storage_archive=True
    )


# ==================== URL SANITIZATION TESTS ====================

@patch('requests.head')
def test_sanitize_url_valid_youtube(mock_head, ingestion):
    """Valid YouTube URL should pass validation"""
    # Mock response for HTTP HEAD reachability check
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_head.return_value = mock_response

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = ingestion.sanitize_url(url)
    
    assert result["status"] == "valid"
    assert result["domain"] == "www.youtube.com"
    assert result["protocol"] == "https"
    assert result["url"] == url
    mock_head.assert_called_once()


@patch('requests.head')
def test_sanitize_url_injection_attempt(mock_head, ingestion):
    """URL with ; & | should be rejected"""
    malicious_urls = [
        "https://youtube.com/watch?v=dQw4w9WgXcQ; rm -rf /",
        "https://youtube.com/watch?v=dQw4w9WgXcQ&other_param=1|cat /etc/passwd",
        "https://youtube.com/watch?v=dQw4w9WgXcQ`id`"
    ]
    
    for url in malicious_urls:
        with pytest.raises(ValueError) as exc_info:
            ingestion.sanitize_url(url)
        assert "dangerous character" in str(exc_info.value)
    
    # Injection checks should fail before any HTTP HEAD request is made
    mock_head.assert_not_called()


@patch('requests.head')
def test_sanitize_url_sql_and_code_injection(mock_head, ingestion):
    """SQL and code injection patterns should be rejected"""
    malicious_urls = [
        "https://youtube.com/watch?v=dQw4w9WgXcQ' UNION SELECT * FROM users--",
        "https://youtube.com/watch?v=dQw4w9WgXcQ#",
        "https://youtube.com/watch?v=eval(x)",
        "https://youtube.com/watch?v=system('ls')",
        "https://youtube.com/watch?v=SELECT",
        "https://youtube.com/watch?v=drop",
        "https://youtube.com/watch?v=union"
    ]
    
    for url in malicious_urls:
        with pytest.raises(ValueError) as exc_info:
            ingestion.sanitize_url(url)
        assert "injection" in str(exc_info.value) or "dangerous character" in str(exc_info.value)
        
    mock_head.assert_not_called()


@patch('requests.head')
def test_sanitize_url_invalid_domain(mock_head, ingestion):
    """Non-whitelisted domain should be rejected"""
    url = "https://facebook.com/video"
    with pytest.raises(ValueError) as exc_info:
        ingestion.sanitize_url(url)
    assert "not whitelisted" in str(exc_info.value)
    mock_head.assert_not_called()


@patch('requests.head')
def test_sanitize_url_unreachable(mock_head, ingestion):
    """Unreachable URL should raise ValueError"""
    # Mock requests.head throwing Timeout exception
    mock_head.side_effect = requests.Timeout("Connection timeout")
    
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    with pytest.raises(ValueError) as exc_info:
        ingestion.sanitize_url(url)
    assert "Connection timeout verifying URL reachability" in str(exc_info.value)


# ==================== SOURCE DETECTION TESTS ====================

@patch('requests.head')
def test_detect_source_youtube(mock_head, ingestion):
    """YouTube URL should return 'youtube'"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_head.return_value = mock_response

    youtube_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ"
    ]
    
    for url in youtube_urls:
        assert ingestion.detect_source(url) == "youtube"


@patch('requests.head')
def test_detect_source_google_drive(mock_head, ingestion):
    """Google Drive URL should return 'google_drive'"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_head.return_value = mock_response

    gdrive_urls = [
        "https://drive.google.com/file/d/1ABC2DEF/view",
        "https://drive.google.com/open?id=1ABC2DEF",
        "https://drive.google.com/uc?id=1ABC2DEF"
    ]
    
    for url in gdrive_urls:
        assert ingestion.detect_source(url) == "google_drive"


# ==================== FILE ID EXTRACTION TESTS ====================

def test_extract_gdrive_file_id_format1(ingestion):
    """Extract ID from /file/d/{ID}/view format"""
    url = "https://drive.google.com/file/d/1t_2e3s_4t_5i_6d_7I_8D_9F_0I_Le/view?usp=sharing"
    file_id = ingestion._extract_gdrive_file_id(url)
    assert file_id == "1t_2e3s_4t_5i_6d_7I_8D_9F_0I_Le"


def test_extract_gdrive_file_id_format2(ingestion):
    """Extract ID from ?id={ID} format"""
    url = "https://drive.google.com/open?id=1t_2e3s_4t_5i_6d_7I_8D_9F_0I_Le"
    file_id = ingestion._extract_gdrive_file_id(url)
    assert file_id == "1t_2e3s_4t_5i_6d_7I_8D_9F_0I_Le"

    url_uc = "https://drive.google.com/uc?id=1t_2e3s_4t_5i_6d_7I_8D_9F_0I_Le"
    file_id_uc = ingestion._extract_gdrive_file_id(url_uc)
    assert file_id_uc == "1t_2e3s_4t_5i_6d_7I_8D_9F_0I_Le"


def test_extract_gdrive_file_id_invalid(ingestion):
    """Invalid Google Drive URL should raise ValueError"""
    url = "https://drive.google.com/invalid_path"
    with pytest.raises(ValueError) as exc_info:
        ingestion._extract_gdrive_file_id(url)
    assert "Could not extract Google Drive file ID" in str(exc_info.value)


# ==================== VIDEO DURATION VALIDATION TESTS ====================

@patch('cv2.VideoCapture')
@patch('os.path.exists')
def test_validate_duration_short_video(mock_exists, mock_video_capture, ingestion):
    """Video < 60 min should pass"""
    mock_exists.return_value = True
    
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # FPS = 30, frame count = 90000 (3000 seconds = 50 min)
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == 5 else 90000.0 if prop == 7 else 0.0
    mock_video_capture.return_value = mock_cap
    
    # 5 is CAP_PROP_FPS, 7 is CAP_PROP_FRAME_COUNT in OpenCV (actually cv2.CAP_PROP_FPS is 5, cv2.CAP_PROP_FRAME_COUNT is 7)
    with patch('cv2.CAP_PROP_FPS', 5), patch('cv2.CAP_PROP_FRAME_COUNT', 7):
        assert ingestion.validate_video_duration("dummy_path.mp4", max_duration=3600) is True
        mock_cap.release.assert_called_once()


@patch('cv2.VideoCapture')
@patch('os.path.exists')
def test_validate_duration_long_video(mock_exists, mock_video_capture, ingestion):
    """Video > 60 min should raise ValueError"""
    mock_exists.return_value = True
    
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # FPS = 30, frame count = 120000 (4000 seconds = 66.7 min)
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == 5 else 120000.0 if prop == 7 else 0.0
    mock_video_capture.return_value = mock_cap
    
    with patch('cv2.CAP_PROP_FPS', 5), patch('cv2.CAP_PROP_FRAME_COUNT', 7):
        with pytest.raises(ValueError) as exc_info:
            ingestion.validate_video_duration("dummy_path.mp4", max_duration=3600)
        assert "Video too long" in str(exc_info.value)
        mock_cap.release.assert_called_once()


@patch('cv2.VideoCapture')
@patch('os.path.exists')
def test_validate_duration_corrupted(mock_exists, mock_video_capture, ingestion):
    """Corrupted/Unreadable video should raise ValueError"""
    mock_exists.return_value = True
    
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_video_capture.return_value = mock_cap
    
    with pytest.raises(ValueError) as exc_info:
        ingestion.validate_video_duration("corrupted.mp4")
    assert "Corrupted or unreadable" in str(exc_info.value)


# ==================== METADATA EXTRACTION TESTS ====================

@patch('cv2.VideoCapture')
@patch('os.path.exists')
@patch('os.path.getsize')
def test_extract_metadata_creates_files(mock_getsize, mock_exists, mock_video_capture, ingestion):
    """Metadata JSON and MD files should be created correctly"""
    mock_exists.return_value = True
    mock_getsize.return_value = 10485760  # 10 MB
    
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    
    # Mocking CAP_PROP_FPS(5), CAP_PROP_FRAME_COUNT(7), CAP_PROP_FRAME_WIDTH(3), CAP_PROP_FRAME_HEIGHT(4)
    # FPS = 30.0, Frames = 36000 (1200 seconds = 20 min), Width = 1920, Height = 1080
    def mock_get(prop):
        if prop == 5: return 30.0
        if prop == 7: return 36000.0
        if prop == 3: return 1920.0
        if prop == 4: return 1080.0
        return 0.0
        
    mock_cap.get.side_effect = mock_get
    mock_video_capture.return_value = mock_cap
    
    with patch('cv2.CAP_PROP_FPS', 5), \
         patch('cv2.CAP_PROP_FRAME_COUNT', 7), \
         patch('cv2.CAP_PROP_FRAME_WIDTH', 3), \
         patch('cv2.CAP_PROP_FRAME_HEIGHT', 4):
         
        video_path = str(ingestion.output_dir / "lecture_001.mp4")
        
        # Call function
        json_path_str = ingestion.extract_and_save_metadata(
            video_path=video_path,
            source_url="https://youtube.com/watch?v=lecture001",
            source_type="youtube"
        )
        
        json_path = Path(json_path_str)
        md_path = json_path.with_suffix('.md')
        
        # Check files exist
        assert json_path.exists()
        assert md_path.exists()
        
        # Verify JSON contents
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        assert data["video_info"]["video_id"] == "lecture_001"
        assert data["video_info"]["duration_formatted"] == "20:00"
        assert data["video_info"]["resolution"] == "1920x1080"
        assert data["video_info"]["file_size_mb"] == 10.0
        assert data["ingestion_info"]["status"] == "ready_for_processing"
        
        # Verify MD content structure
        with open(md_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
        assert "# Video Metadata - lecture_001" in md_text
        assert "- **Resolution:** 1920x1080" in md_text
        assert "- **File Size:** 10.0 MB" in md_text
