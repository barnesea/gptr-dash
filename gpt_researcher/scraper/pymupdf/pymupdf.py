import os
import requests
import tempfile
import time
from urllib.parse import urlparse
from langchain_community.document_loaders import PyMuPDFLoader


class PyMuPDFScraper:

    def __init__(
        self,
        link,
        session=None,
        *,
        connect_timeout_seconds: float = 5.0,
        total_timeout_seconds: float = 30.0,
        max_download_bytes: int = 32 * 1024 * 1024,
    ):
        """
        Initialize the scraper with a link and an optional session.

        Args:
          link (str): The URL or local file path of the PDF document.
          session (requests.Session, optional): An optional session for making HTTP requests.
        """
        self.link = link
        self.session = session
        self.connect_timeout_seconds = max(
            0.25, float(connect_timeout_seconds)
        )
        self.total_timeout_seconds = max(0.5, float(total_timeout_seconds))
        self.max_download_bytes = max(1024, int(max_download_bytes))
        self.last_error_type = ""
        self.last_error_detail = ""

    def _download_pdf(self):
        """Download a PDF under one deadline shared by SSL fallback attempts."""
        deadline = time.monotonic() + self.total_timeout_seconds
        request = self.session.get if self.session is not None else requests.get
        response = None
        for verify in (True, False):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise requests.exceptions.Timeout("PDF download deadline expired")
            try:
                response = request(
                    self.link,
                    timeout=(
                        min(self.connect_timeout_seconds, remaining),
                        max(0.25, remaining),
                    ),
                    stream=True,
                    verify=verify,
                )
                response.raise_for_status()
                break
            except requests.exceptions.SSLError:
                if not verify:
                    raise
                import logging
                logging.getLogger(__name__).warning(
                    "SSL verification failed for %s, retrying without verification",
                    self.link,
                )
        if response is None:
            raise requests.exceptions.RequestException("PDF download failed")

        content_length = getattr(response, "headers", {}).get("Content-Length")
        if content_length:
            try:
                declared_bytes = int(content_length)
            except (TypeError, ValueError):
                declared_bytes = 0
            if declared_bytes > self.max_download_bytes:
                close_response = getattr(response, "close", None)
                if callable(close_response):
                    close_response()
                raise ValueError("pdf_size_limit")

        downloaded = 0
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as temp_file:
            temp_filename = temp_file.name
            try:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if time.monotonic() >= deadline:
                        raise requests.exceptions.Timeout(
                            "PDF download deadline expired"
                        )
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > self.max_download_bytes:
                        raise ValueError("pdf_size_limit")
                    temp_file.write(chunk)
            except Exception:
                try:
                    os.remove(temp_filename)
                except OSError:
                    pass
                raise
            finally:
                close_response = getattr(response, "close", None)
                if callable(close_response):
                    close_response()
        return temp_filename

    def is_url(self) -> bool:
        """
        Check if the provided `link` is a valid URL.

        Returns:
          bool: True if the link is a valid URL, False otherwise.
        """
        try:
            result = urlparse(self.link)
            return all([result.scheme, result.netloc])  # Check for valid scheme and network location
        except Exception:
            return False

    def scrape(self) -> tuple[str, list[str], str]:
        """
        The `scrape` function uses PyMuPDFLoader to load a document from the provided link (either URL or local file)
        and returns the document as a string.

        Returns:
          str: A string representation of the loaded document.
        """
        try:
            if self.is_url():
                temp_filename = self._download_pdf()
                # Always clean up the downloaded temp file, even if loading fails
                # (PyMuPDFLoader.load() can raise on a malformed/partial PDF).
                try:
                    loader = PyMuPDFLoader(temp_filename)
                    try:
                        doc = loader.load()
                    except Exception:
                        self.last_error_type = "parse_failure"
                        raise
                finally:
                    try:
                        os.remove(temp_filename)
                    except OSError:
                        pass
            else:
                loader = PyMuPDFLoader(self.link)
                doc = loader.load()

            # Extract the content, image (if any), and title from the document.
            image = []
            # Retrieve content from ALL pages to ensure PDFs with cover pages pass validation.
            content = "\n".join(page.page_content for page in doc)
            title = doc[0].metadata.get("title", "") if doc else ""
            return content, image, title

        except requests.exceptions.Timeout:
            self.last_error_type = "download_timeout"
            self.last_error_detail = (
                f"exceeded {self.total_timeout_seconds:g}s total deadline"
            )
            print(f"Download timed out. Please check the link : {self.link}")
            return "", [], ""
        except requests.exceptions.HTTPError as error:
            status = getattr(error.response, "status_code", None)
            self.last_error_type = (
                "blocked" if status in {401, 403, 429} else "download_error"
            )
            self.last_error_detail = f"HTTP {status}" if status else str(error)
            return "", [], ""
        except ValueError as error:
            if str(error) == "pdf_size_limit":
                self.last_error_type = "size_limit"
                self.last_error_detail = (
                    f"exceeded {self.max_download_bytes} byte limit"
                )
                return "", [], ""
            self.last_error_type = self.last_error_type or "parse_failure"
            self.last_error_detail = str(error)
            return "", [], ""
        except Exception as e:
            self.last_error_type = self.last_error_type or "parse_failure"
            self.last_error_detail = str(e)
            print(f"Error loading PDF : {self.link} {e}")
            return "", [], ""
