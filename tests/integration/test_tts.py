import base64
from unittest.mock import MagicMock, patch

from src.tools.tts import OpenAITTS


class TestOpenAITTS:
    """Test suite for the OpenAITTS class."""

    def test_initialization(self):
        tts = OpenAITTS(api_key="key", model="tts-1", voice="alloy", host="http://api")
        assert tts.api_key == "key"
        assert tts.model == "tts-1"
        assert tts.voice == "alloy"
        assert tts.host == "http://api"

    @patch("src.tools.tts.requests.post")
    def test_text_to_speech_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"audio_data"
        mock_post.return_value = mock_response

        tts = OpenAITTS(api_key="key")
        result = tts.text_to_speech("Hello")

        assert result["success"] is True
        assert base64.b64decode(result["audio_data"]) == b"audio_data"
        args, _ = mock_post.call_args
        assert args[0] == "https://api.openai.com/v1/audio/speech"

    @patch("src.tools.tts.requests.post")
    def test_text_to_speech_api_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "bad request"
        mock_post.return_value = mock_response

        tts = OpenAITTS(api_key="key")
        result = tts.text_to_speech("Hello")

        assert result["success"] is False
        assert result["error"] == "bad request"

    @patch("src.tools.tts.requests.post")
    def test_text_to_speech_request_exception(self, mock_post):
        mock_post.side_effect = Exception("network")
        tts = OpenAITTS(api_key="key")
        result = tts.text_to_speech("Hello")

        assert result["success"] is False
        assert result["audio_data"] is None
