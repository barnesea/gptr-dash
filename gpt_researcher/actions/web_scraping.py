from typing import Any
from colorama import Fore, Style

from gpt_researcher.utils.workers import WorkerPool
from ..scraper import Scraper
from ..config.config import Config
from ..utils.logger import get_formatted_logger

logger = get_formatted_logger()


async def scrape_urls(
    urls,
    cfg: Config,
    worker_pool: WorkerPool,
    *,
    deep_research: bool = False,
    include_failures: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]
]:
    """
    Scrapes the urls
    Args:
        urls: List of urls
        cfg: Config (optional)

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]]]: tuple containing scraped content and images

    """
    scraped_data = []
    images = []
    user_agent = (
        cfg.user_agent
        if cfg
        else "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    )

    scraper = None
    try:
        scraper = Scraper(
            urls,
            user_agent,
            cfg.scraper,
            worker_pool=worker_pool,
            pdf_connect_timeout_seconds=(
                getattr(
                    cfg,
                    "deep_research_pdf_connect_timeout_seconds",
                    3.0,
                )
                if deep_research
                else 5.0
            ),
            pdf_total_timeout_seconds=(
                getattr(
                    cfg,
                    "deep_research_pdf_total_timeout_seconds",
                    8.0,
                )
                if deep_research
                else 30.0
            ),
            pdf_max_download_bytes=getattr(
                cfg,
                "deep_research_pdf_max_bytes",
                32 * 1024 * 1024,
            ),
        )
        scraped_data = await scraper.run()
        for item in scraped_data:
            if 'image_urls' in item:
                images.extend(item['image_urls'])
    except Exception as e:
        print(f"{Fore.RED}Error in scrape_urls: {e}{Style.RESET_ALL}")
    finally:
        # Close the requests.Session so its underlying connection pool (and the
        # sockets it keeps alive) is released
        if scraper is not None and getattr(scraper, "session", None) is not None:
            scraper.session.close()

    failures = list(getattr(scraper, "failures", []) or [])
    if include_failures:
        return scraped_data, images, failures
    return scraped_data, images


async def filter_urls(urls: list[str], config: Config) -> list[str]:
    """
    Filter URLs based on configuration settings.

    Args:
        urls (list[str]): List of URLs to filter.
        config (Config): Configuration object.

    Returns:
        list[str]: Filtered list of URLs.
    """
    filtered_urls = []
    for url in urls:
        # Add your filtering logic here
        # For example, you might want to exclude certain domains or URL patterns
        if not any(excluded in url for excluded in config.excluded_domains):
            filtered_urls.append(url)
    return filtered_urls

async def extract_main_content(html_content: str) -> str:
    """
    Extract the main content from HTML.

    Args:
        html_content (str): Raw HTML content.

    Returns:
        str: Extracted main content.
    """
    # Implement content extraction logic here
    # This could involve using libraries like BeautifulSoup or custom parsing logic
    # For now, we'll just return the raw HTML as a placeholder
    return html_content

async def process_scraped_data(scraped_data: list[dict[str, Any]], config: Config) -> list[dict[str, Any]]:
    """
    Process the scraped data to extract and clean the main content.

    Args:
        scraped_data (list[dict[str, Any]]): List of dictionaries containing scraped data.
        config (Config): Configuration object.

    Returns:
        list[dict[str, Any]]: Processed scraped data.
    """
    processed_data = []
    for item in scraped_data:
        if item['status'] == 'success':
            main_content = await extract_main_content(item['content'])
            processed_data.append({
                'url': item['url'],
                'content': main_content,
                'status': 'success'
            })
        else:
            processed_data.append(item)
    return processed_data
