from __future__ import annotations

from typing import Any

import pytest
from livekit.agents.llm import ToolError

from browser import BrowserError, BrowserManager
from tools import BrowserTools


class FakeBrowser:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def open_url(self, url: str) -> dict[str, str]:
        self.calls.append(("open_url", url))
        return {"url": url, "title": "Test"}

    async def click(self, target: str) -> dict[str, str]:
        self.calls.append(("click", target))
        return {"url": "https://example.com", "title": "Clicked"}

    async def type_text(self, target: str, text: str) -> dict[str, str]:
        self.calls.append(("type_text", target, text))
        return {"target": target, "url": "https://example.com"}

    async def press_key(self, key: str) -> dict[str, str]:
        self.calls.append(("press_key", key))
        return {"key": key, "url": "https://example.com"}


def test_browser_tools_expose_expected_names() -> None:
    tools = BrowserTools(FakeBrowser())  # type: ignore[arg-type]

    assert {tool.info.name for tool in tools.tools} == {
        "click",
        "go_back",
        "inspect_page",
        "open_url",
        "press_key",
        "read_page",
        "scroll",
        "search_the_web",
        "take_screenshot",
        "type_text",
    }


@pytest.mark.asyncio
async def test_search_opens_encoded_duckduckgo_query() -> None:
    browser = FakeBrowser()
    tools = BrowserTools(browser)  # type: ignore[arg-type]

    result = await tools.search_the_web("livekit voice agent")

    assert result["url"] == "https://duckduckgo.com/?q=livekit+voice+agent"
    assert browser.calls == [("open_url", result["url"])]


@pytest.mark.asyncio
async def test_consequential_click_requires_matching_confirmation() -> None:
    browser = FakeBrowser()
    tools = BrowserTools(browser)  # type: ignore[arg-type]

    first_result = await tools.click("Delete account")

    assert first_result["status"] == "confirmation_required"
    assert browser.calls == []

    with pytest.raises(ToolError, match="exact action"):
        await tools.click("Delete all accounts", confirmed=True)

    final_result = await tools.click("Delete account", confirmed=True)

    assert final_result["title"] == "Clicked"
    assert browser.calls == [("click", "Delete account")]


@pytest.mark.asyncio
async def test_sensitive_fields_are_never_filled() -> None:
    browser = FakeBrowser()
    tools = BrowserTools(browser)  # type: ignore[arg-type]

    with pytest.raises(ToolError, match="cannot enter passwords"):
        await tools.type_text("Password", "secret")

    assert browser.calls == []


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "http://127.0.0.1:8080",
        "https://user:secret@example.com",
    ],
)
def test_url_validation_rejects_unsafe_addresses(url: str) -> None:
    with pytest.raises(BrowserError):
        BrowserManager._validate_url(url)


def test_url_validation_normalizes_whitespace() -> None:
    assert (
        BrowserManager._validate_url("  https://example.com/docs  ")
        == "https://example.com/docs"
    )
