import pytest
from modules.ingest import detect_source, _extract_gdrive_file_id, sanitize_url


class TestDetectSource:
    def test_standard_youtube_url(self):
        assert detect_source("https://www.youtube.com/watch?v=abc123") == "youtube"

    def test_short_youtube_url(self):
        assert detect_source("https://youtu.be/abc123") == "youtube"

    def test_youtube_url_with_playlist(self):
        assert detect_source("https://youtu.be/abc123?list=PLxyz") == "youtube"

    def test_gdrive_url(self):
        assert detect_source("https://drive.google.com/file/d/abc123/view") == "gdrive"

    def test_unknown_url(self):
        assert detect_source("https://vimeo.com/12345") == "unknown"

    def test_empty_string(self):
        assert detect_source("") == "unknown"


class TestExtractGdriveFileId:
    def test_standard_share_link(self):
        url = "https://drive.google.com/file/d/1A2B3C4D5E/view?usp=sharing"
        assert _extract_gdrive_file_id(url) == "1A2B3C4D5E"

    def test_open_id_link(self):
        url = "https://drive.google.com/open?id=1A2B3C4D5E"
        assert _extract_gdrive_file_id(url) == "1A2B3C4D5E"

    def test_malformed_url_returns_none(self):
        url = "https://drive.google.com/folder/notavalidlink"
        assert _extract_gdrive_file_id(url) is None

    def test_same_file_id_is_deterministic(self):
        url = "https://drive.google.com/file/d/1A2B3C4D5E/view"
        assert _extract_gdrive_file_id(url) == _extract_gdrive_file_id(url)


class TestSanitizeUrl:
    def test_valid_youtube_url_passes(self):
        sanitize_url("https://www.youtube.com/watch?v=abc123")  # should not raise

    def test_valid_youtube_short_url_passes(self):
        sanitize_url("https://youtu.be/abc123")

    def test_valid_gdrive_url_passes(self):
        sanitize_url("https://drive.google.com/file/d/abc123/view")

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError):
            sanitize_url("")

    def test_none_rejected(self):
        with pytest.raises(ValueError):
            sanitize_url(None)

    def test_non_http_scheme_rejected(self):
        with pytest.raises(ValueError):
            sanitize_url("ftp://youtube.com/watch?v=abc123")

    def test_javascript_scheme_rejected(self):
        with pytest.raises(ValueError):
            sanitize_url("javascript:alert(1)")

    def test_unsupported_domain_rejected(self):
        with pytest.raises(ValueError):
            sanitize_url("https://vimeo.com/12345")

    def test_dangerous_semicolon_rejected(self):
        with pytest.raises(ValueError):
            sanitize_url("https://youtube.com/watch?v=abc;rm -rf /")

    def test_dangerous_backtick_rejected(self):
        with pytest.raises(ValueError):
            sanitize_url("https://youtube.com/watch?v=`whoami`")

    def test_dangerous_pipe_rejected(self):
        with pytest.raises(ValueError):
            sanitize_url("https://youtube.com/watch?v=abc|cat /etc/passwd")

    def test_oversized_url_rejected(self):
        huge_url = "https://youtube.com/watch?v=" + "a" * 3000
        with pytest.raises(ValueError):
            sanitize_url(huge_url)

    def test_subdomain_of_allowed_domain_passes(self):
        # e.g. m.youtube.com should still be accepted
        sanitize_url("https://m.youtube.com/watch?v=abc123")

    def test_lookalike_domain_rejected(self):
        # youtube.com.evil.com is NOT youtube.com — must be rejected
        with pytest.raises(ValueError):
            sanitize_url("https://youtube.com.evil-site.com/watch?v=abc123")