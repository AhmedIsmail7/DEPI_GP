from modules.ingest import detect_source, _extract_gdrive_file_id


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