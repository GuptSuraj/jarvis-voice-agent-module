from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar
from urllib.parse import urlencode

from livekit.agents import function_tool
from livekit.agents.llm import FunctionTool, ToolError

from browser import BrowserError, BrowserManager

_T = TypeVar("_T")

_CONSEQUENTIAL_TARGET = re.compile(
    r"\b(buy|purchase|order|checkout|pay|send|post|publish|submit|apply|"
    r"register|sign[ -]?up|book|reserve|delete|remove|unsubscribe|transfer|"
    r"confirm|approve)\b",
    re.IGNORECASE,
)
_SENSITIVE_TARGET = re.compile(
    r"\b(password|passcode|one[ -]?time|otp|two[ -]?factor|2fa|card|cvv|cvc|"
    r"expiry|social security|ssn|iban|routing|account number)\b",
    re.IGNORECASE,
)
_SEARCH_TARGET = re.compile(r"\b(search|find|query|look[ -]?up|filter)\b", re.I)


class BrowserTools:
    """Expose one session-scoped :class:`BrowserManager` to a LiveKit agent."""

    def __init__(self, browser: BrowserManager) -> None:
        self._browser = browser
        self._pending_confirmation: tuple[str, ...] | None = None
        self.tools: list[FunctionTool] = [
            function_tool(self.open_url),
            function_tool(self.search_the_web),
            function_tool(self.inspect_page),
            function_tool(self.read_page),
            function_tool(self.go_back),
            function_tool(self.click),
            function_tool(self.type_text),
            function_tool(self.scroll),
            function_tool(self.press_key),
            function_tool(self.take_screenshot),
        ]

    async def _run(self, operation: Callable[[], Awaitable[_T]]) -> _T:
        try:
            return await operation()
        except BrowserError as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            raise ToolError("The browser operation failed unexpectedly.") from exc

    def _clear_confirmation(self) -> None:
        self._pending_confirmation = None

    def _confirmation_result(
        self,
        action: tuple[str, ...],
        description: str,
        *,
        confirmed: bool,
    ) -> dict[str, Any] | None:
        if not confirmed:
            self._pending_confirmation = action
            return {
                "status": "confirmation_required",
                "action": description,
                "message": (
                    "Explain this exact action to the user and ask for confirmation. "
                    "Only retry it with confirmed=true after the user agrees."
                ),
            }

        if self._pending_confirmation != action:
            raise ToolError(
                "That exact action is not awaiting confirmation. Inspect the page and "
                "ask the user before trying again."
            )

        self._pending_confirmation = None
        return None

    async def open_url(self, url: str) -> dict[str, str]:
        """Open a complete HTTP or HTTPS URL in the agent-controlled browser.

        Use this when the user names a website, service, domain, or link. Open the
        official site directly instead of using a general web search.

        Args:
            url: Complete web address, including http:// or https://.
        """
        self._clear_confirmation()
        return await self._run(lambda: self._browser.open_url(url))

    async def search_the_web(self, query: str) -> dict[str, str]:
        """Open DuckDuckGo results for a general internet search.

        Use this only when the user did not specify a website or service. Inspect or
        read the resulting page before answering.

        Args:
            query: Concise search query containing the important terms.
        """
        if not query.strip():
            raise ToolError("A search query is required.")
        self._clear_confirmation()
        url = f"https://duckduckgo.com/?{urlencode({'q': query.strip()})}"
        return await self._run(lambda: self._browser.open_url(url))

    async def inspect_page(self, max_chars: int = 8_000) -> dict[str, object]:
        """Read the page and list its visible interactive elements.

        Call this before clicking or typing. Use returned element names as targets.

        Args:
            max_chars: Maximum number of page-text characters to return.
        """
        max_chars = max(500, min(max_chars, 12_000))
        return await self._run(lambda: self._browser.inspect_page(max_chars=max_chars))

    async def read_page(self, max_chars: int = 12_000) -> dict[str, str | bool]:
        """Read the visible text of the current page.

        Args:
            max_chars: Maximum number of page-text characters to return.
        """
        max_chars = max(500, min(max_chars, 20_000))
        return await self._run(lambda: self._browser.read_page(max_chars=max_chars))

    async def go_back(self) -> dict[str, str]:
        """Navigate the current browser tab to its previous page."""
        self._clear_confirmation()
        return await self._run(self._browser.go_back)

    async def click(
        self,
        target: str,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        """Click a visible control identified by a name from inspect_page.

        Consequential actions return a confirmation request without clicking. After
        the user explicitly agrees, retry the exact action with confirmed=true.

        Args:
            target: Accessible name of the button, link, or other control.
            confirmed: True only after the user confirms this exact action.
        """
        action = ("click", target)
        is_consequential = bool(
            _CONSEQUENTIAL_TARGET.search(target) and not _SEARCH_TARGET.search(target)
        )
        if is_consequential:
            result = self._confirmation_result(
                action,
                f"click {target!r}",
                confirmed=confirmed,
            )
            if result is not None:
                return result
        else:
            self._clear_confirmation()

        return await self._run(lambda: self._browser.click(target))

    async def type_text(
        self,
        target: str,
        text: str,
        submit: bool = False,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        """Type user-provided text into a field and optionally press Enter.

        Never use this for passwords, payment details, or one-time codes. Submitting
        non-search fields requires explicit user confirmation.

        Args:
            target: Accessible field name returned by inspect_page.
            text: Text supplied or approved by the user.
            submit: Press Enter after filling the field.
            confirmed: True only after the user confirms this exact submission.
        """
        if _SENSITIVE_TARGET.search(target):
            raise ToolError(
                "I cannot enter passwords, payment details, or authentication codes. "
                "The user must enter that information themselves."
            )

        action = ("type_text", target, text, str(submit))
        needs_confirmation = submit and not _SEARCH_TARGET.search(target)
        if needs_confirmation:
            result = self._confirmation_result(
                action,
                f"submit text in {target!r}",
                confirmed=confirmed,
            )
            if result is not None:
                return result
        else:
            self._clear_confirmation()

        result: dict[str, Any] = await self._run(
            lambda: self._browser.type_text(target, text)
        )
        if submit:
            await self._run(lambda: self._browser.press_key("Enter"))
            result["submitted"] = True
        return result

    async def scroll(self, direction: str) -> dict[str, str]:
        """Scroll the current page up or down.

        Args:
            direction: Either "up" or "down".
        """
        if direction not in {"up", "down"}:
            raise ToolError("Scroll direction must be 'up' or 'down'.")
        return await self._run(lambda: self._browser.scroll(direction))

    async def press_key(self, key: str) -> dict[str, str]:
        """Press a safe navigation key in the current page.

        Args:
            key: Enter, Escape, Tab, an arrow key, or Backspace.
        """
        return await self._run(lambda: self._browser.press_key(key))

    async def take_screenshot(self) -> dict[str, str | int | bool]:
        """Capture the current viewport for diagnostics."""
        return await self._run(self._browser.take_screenshot)
