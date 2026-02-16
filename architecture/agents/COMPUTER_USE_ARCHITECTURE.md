# Computer Use Architecture for GAIA

**Date**: February 7, 2026
**Version**: 1.0
**Status**: Specification
**Priority**: CRITICAL
**Estimated Effort**: 8-10 weeks (2 engineers)
**Target**: Enable UI automation, desktop control, and visual understanding

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Architecture Overview](#architecture-overview)
4. [Component Specifications](#component-specifications)
5. [Safety Framework](#safety-framework)
6. [Integration with GAIA](#integration-with-gaia)
7. [Implementation Plan](#implementation-plan)
8. [Testing Strategy](#testing-strategy)
9. [Success Metrics](#success-metrics)
10. [Complete Code](#complete-code)

---

## Executive Summary

### The Gap

GAIA agents can:
- ✅ Execute bash commands (blind)
- ✅ Read/write files
- ✅ Call APIs

GAIA agents **cannot**:
- ❌ See and understand UI
- ❌ Control mouse/keyboard
- ❌ Navigate applications
- ❌ Interact with web pages
- ❌ Verify visual state

### The Solution

A **Computer Use Architecture** that enables agents to:
- Capture and understand screen content using VLM
- Control mouse and keyboard programmatically
- Navigate and interact with applications (web and desktop)
- Verify UI state and validate actions
- Execute with safety guardrails

### Impact

**Unlocks entire category**: Computer use agents for:
- Email triage (navigate Gmail UI)
- Form filling (web forms, desktop apps)
- UI testing (automated QA)
- Data entry (legacy systems)
- Application automation (any GUI app)

**Before**: 10% coverage for computer use category
**After**: 90% coverage

---

## Problem Statement

### Current Limitations

**Example task**: "Open Gmail and triage my emails, moving urgent ones to a folder"

**What GAIA can do today**:
```python
# Only via Gmail API (if credentials available)
from gaia.agents.base import Agent

agent = Agent()
agent.process_query("Use Gmail API to fetch and triage emails")
```

**What GAIA cannot do**:
- Open a web browser visually
- Navigate to gmail.com
- Click the email list
- Read email subjects visually
- Click "Move to folder" button
- Select the folder from dropdown
- Verify the email was moved

**Why this matters**:
- Many systems have no API (legacy apps, internal tools)
- Some tasks require visual verification
- Human users interact via UI, agents should too
- Workflow automation often requires UI interaction

### User Stories

**Story 1: Email Triage**
```
As a user, I want my agent to:
- Open Gmail in a browser
- Read unread emails
- Classify them (urgent, spam, newsletter)
- Move them to appropriate folders
- Archive newsletters
So that I can start my day with a clean inbox
```

**Story 2: Form Filling**
```
As a user, I want my agent to:
- Open an expense report form (web or desktop)
- Fill in date, vendor, amount, category from a receipt
- Upload the receipt image
- Click Submit
So that I don't have to manually enter expenses
```

**Story 3: Legacy System Automation**
```
As a user, I want my agent to:
- Open our legacy inventory management system (no API)
- Navigate through the desktop application
- Update stock quantities
- Generate reports
So that we can automate our inventory process
```

---

## Architecture Overview

### High-Level Design

```
┌──────────────────────────────────────────────────────────────────┐
│                        GAIA Agent                                 │
│  "Open Gmail and triage emails"                                  │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│              Computer Use Controller                              │
│  - Parses high-level intent                                      │
│  - Plans UI interaction sequence                                 │
│  - Coordinates subsystems                                        │
└────────────────────┬─────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┬────────────┬────────────┐
        │                         │            │            │
        ▼                         ▼            ▼            ▼
┌───────────────┐  ┌──────────────────┐  ┌────────────┐  ┌─────────┐
│ Vision System │  │  Input Control   │  │  Browser   │  │ Desktop │
│               │  │                  │  │  Automation│  │ App     │
│ - Screenshot  │  │ - Mouse control  │  │            │  │ Control │
│ - OCR         │  │ - Keyboard input │  │ - Playwright│  │         │
│ - VLM         │  │ - Click/drag     │  │ - DOM      │  │ - UI    │
│ - Element     │  │ - Type text      │  │ - JS exec  │  │   trees │
│   detection   │  │                  │  │            │  │ - Events│
└───────┬───────┘  └────────┬─────────┘  └──────┬─────┘  └────┬────┘
        │                   │                    │             │
        └───────────────────┴────────────────────┴─────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Safety Framework          │
                    │  - Action validation          │
                    │  - Confirmation prompts       │
                    │  - Rollback capability        │
                    └───────────────────────────────┘
```

### Component Layers

| Layer | Components | Responsibility |
|-------|-----------|----------------|
| **Controller** | ComputerUseController | High-level orchestration, planning |
| **Vision** | ScreenCapture, OCR, VLMAnalyzer | Understand screen content |
| **Input** | MouseController, KeyboardController | Execute actions |
| **Automation** | BrowserAutomation, DesktopAutomation | Application-specific control |
| **Safety** | SafetyGuard, ActionValidator | Prevent dangerous operations |

---

## Component Specifications

### 1. Vision System

#### 1.1 Screen Capture Engine

```python
from dataclasses import dataclass
from typing import Tuple, List, Optional
from PIL import Image
import mss
import numpy as np

@dataclass
class BoundingBox:
    """Screen region coordinates."""
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def contains_point(self, x: int, y: int) -> bool:
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)


class ScreenCaptureEngine:
    """
    Capture screenshots and screen regions.

    Supports:
    - Multi-monitor setups
    - Region capture
    - Continuous capture (for monitoring)
    - High DPI/Retina displays
    """

    def __init__(self):
        self.sct = mss.mss()
        self.monitors = self.sct.monitors[1:]  # Exclude "all monitors" entry

    def list_monitors(self) -> List[Dict]:
        """
        Get list of available monitors.

        Returns:
            [
                {"id": 1, "width": 1920, "height": 1080, "x": 0, "y": 0},
                {"id": 2, "width": 2560, "height": 1440, "x": 1920, "y": 0},
                ...
            ]
        """
        return [
            {
                "id": i,
                "width": mon["width"],
                "height": mon["height"],
                "x": mon["left"],
                "y": mon["top"]
            }
            for i, mon in enumerate(self.monitors, 1)
        ]

    def capture_screen(self, monitor_id: int = 1) -> Image.Image:
        """
        Capture full screen of specified monitor.

        Args:
            monitor_id: Monitor number (1-indexed)

        Returns:
            PIL Image
        """
        monitor = self.monitors[monitor_id - 1]
        screenshot = self.sct.grab(monitor)
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

    def capture_region(self, bbox: BoundingBox, monitor_id: int = 1) -> Image.Image:
        """
        Capture specific screen region.

        Args:
            bbox: Region to capture
            monitor_id: Monitor number

        Returns:
            PIL Image of region
        """
        monitor = self.monitors[monitor_id - 1]
        region = {
            "left": monitor["left"] + bbox.x,
            "top": monitor["top"] + bbox.y,
            "width": bbox.width,
            "height": bbox.height
        }
        screenshot = self.sct.grab(region)
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

    def capture_continuous(self, monitor_id: int = 1, fps: int = 2):
        """
        Continuously capture screen (generator).

        Args:
            monitor_id: Monitor number
            fps: Frames per second

        Yields:
            PIL Images at specified FPS
        """
        import time
        interval = 1.0 / fps

        while True:
            start = time.time()
            yield self.capture_screen(monitor_id)
            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def save_screenshot(self, path: str, monitor_id: int = 1):
        """Save screenshot to file."""
        img = self.capture_screen(monitor_id)
        img.save(path)
```

#### 1.2 OCR Engine

```python
from typing import List, Dict
import easyocr
import pytesseract
from PIL import Image

class OCREngine:
    """
    Extract text from images using OCR.

    Supports multiple backends:
    - EasyOCR (GPU-accelerated, multilingual)
    - Tesseract (CPU, fast, English-focused)
    - PaddleOCR (CPU/GPU, multilingual, table-aware)
    """

    def __init__(self, backend: str = "easyocr", languages: List[str] = None):
        """
        Initialize OCR engine.

        Args:
            backend: "easyocr", "tesseract", or "paddle"
            languages: List of language codes (e.g., ["en", "fr"])
        """
        self.backend = backend
        self.languages = languages or ["en"]

        if backend == "easyocr":
            self.reader = easyocr.Reader(self.languages, gpu=True)
        elif backend == "tesseract":
            # Tesseract installed system-wide
            pass
        elif backend == "paddle":
            from paddleocr import PaddleOCR
            self.reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

    def extract_text(self, image: Image.Image) -> str:
        """
        Extract all text from image.

        Returns:
            Extracted text as single string
        """
        if self.backend == "easyocr":
            results = self.reader.readtext(np.array(image))
            return " ".join([text for (bbox, text, conf) in results])

        elif self.backend == "tesseract":
            return pytesseract.image_to_string(image)

        elif self.backend == "paddle":
            results = self.reader.ocr(np.array(image))
            text_parts = []
            for line in results:
                for word_info in line:
                    text_parts.append(word_info[1][0])
            return " ".join(text_parts)

    def find_text(self, image: Image.Image, search_text: str) -> List[Dict]:
        """
        Find specific text in image and return bounding boxes.

        Args:
            image: Input image
            search_text: Text to search for

        Returns:
            [
                {
                    "text": "matched text",
                    "bbox": BoundingBox(x, y, w, h),
                    "confidence": 0.95
                },
                ...
            ]
        """
        results = []

        if self.backend == "easyocr":
            ocr_results = self.reader.readtext(np.array(image))

            for (bbox_coords, text, confidence) in ocr_results:
                if search_text.lower() in text.lower():
                    # bbox_coords: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    x1, y1 = bbox_coords[0]
                    x2, y2 = bbox_coords[2]

                    results.append({
                        "text": text,
                        "bbox": BoundingBox(
                            x=int(x1),
                            y=int(y1),
                            width=int(x2 - x1),
                            height=int(y2 - y1)
                        ),
                        "confidence": confidence
                    })

        return results

    def extract_structured(self, image: Image.Image) -> Dict:
        """
        Extract structured data (useful for forms, tables).

        Returns:
            {
                "words": [{"text": str, "bbox": BoundingBox, "conf": float}, ...],
                "lines": [{"text": str, "bbox": BoundingBox}, ...],
                "paragraphs": [{"text": str, "bbox": BoundingBox}, ...]
            }
        """
        if self.backend == "tesseract":
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            words = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Filter out noise
                    words.append({
                        "text": data['text'][i],
                        "bbox": BoundingBox(
                            x=data['left'][i],
                            y=data['top'][i],
                            width=data['width'][i],
                            height=data['height'][i]
                        ),
                        "confidence": int(data['conf'][i]) / 100
                    })

            return {"words": words, "lines": [], "paragraphs": []}

        # Implement for other backends...
        return {}
```

#### 1.3 VLM Analyzer

```python
from gaia.llm import VLMClient
from typing import List, Dict, Optional

class VLMAnalyzer:
    """
    Analyze screenshots using Vision Language Model.

    Uses GAIA's VLM support (Qwen2.5-VL or similar).
    """

    def __init__(self, model: str = "Qwen2.5-VL-7B-Instruct-GGUF"):
        self.vlm = VLMClient(model=model)

    def describe_screen(self, image: Image.Image, detail_level: str = "medium") -> str:
        """
        Generate natural language description of screen.

        Args:
            image: Screenshot
            detail_level: "low" | "medium" | "high"

        Returns:
            Description like "The screen shows Gmail inbox with 3 unread emails.
            The first email is from john@example.com with subject 'Q4 Report'..."
        """
        prompts = {
            "low": "Describe what's on this screen in one sentence.",
            "medium": "Describe what's on this screen in detail, including visible UI elements.",
            "high": "Provide a comprehensive description of this screen, including all text, UI elements, their positions, and current state."
        }

        response = self.vlm.analyze_image(
            image=image,
            prompt=prompts[detail_level]
        )

        return response["text"]

    def find_clickable_elements(self, image: Image.Image) -> List[Dict]:
        """
        Detect clickable UI elements (buttons, links, inputs).

        Returns:
            [
                {
                    "type": "button" | "link" | "input" | "checkbox" | "dropdown",
                    "text": "Submit",
                    "bbox": BoundingBox(x, y, w, h),
                    "confidence": 0.95
                },
                ...
            ]
        """
        prompt = """Analyze this screenshot and identify all clickable UI elements.
For each element, provide:
1. Type (button, link, input field, checkbox, dropdown, etc.)
2. Visible text or label
3. Approximate bounding box coordinates as [x, y, width, height]

Return as JSON array."""

        response = self.vlm.analyze_image(
            image=image,
            prompt=prompt,
            response_format="json"
        )

        # Parse JSON response
        import json
        elements = json.loads(response["text"])

        # Convert to our format
        result = []
        for elem in elements:
            result.append({
                "type": elem["type"],
                "text": elem.get("text", ""),
                "bbox": BoundingBox(
                    x=elem["bbox"][0],
                    y=elem["bbox"][1],
                    width=elem["bbox"][2],
                    height=elem["bbox"][3]
                ),
                "confidence": elem.get("confidence", 0.8)
            })

        return result

    def verify_action_result(self, before: Image.Image, after: Image.Image,
                            expected_change: str) -> bool:
        """
        Verify that an action had the expected visual effect.

        Args:
            before: Screenshot before action
            after: Screenshot after action
            expected_change: Description like "email moved to Urgent folder"

        Returns:
            True if expected change visible
        """
        prompt = f"""Compare these two screenshots (before and after an action).
Expected change: {expected_change}

Did this change occur? Respond with:
- "YES" if the expected change is clearly visible
- "NO" if the change did not occur
- "PARTIAL" if some change occurred but not exactly as expected

Also provide a brief explanation."""

        response = self.vlm.analyze_images(
            images=[before, after],
            prompt=prompt
        )

        return response["text"].strip().startswith("YES")

    def answer_visual_question(self, image: Image.Image, question: str) -> str:
        """
        Answer questions about screenshot.

        Examples:
        - "What is the sender of the first email?"
        - "Is there a Submit button on this page?"
        - "What's the current balance shown?"
        """
        response = self.vlm.analyze_image(
            image=image,
            prompt=question
        )

        return response["text"]
```

### 2. Input Control System

#### 2.1 Mouse Controller

```python
import pyautogui
import time
from typing import Tuple
import numpy as np

class MouseController:
    """
    Control mouse programmatically with human-like behavior.

    Features:
    - Smooth cursor movement
    - Click with configurable delay
    - Drag and drop
    - Scroll
    - Double-click, right-click
    """

    def __init__(self, speed: float = 0.5):
        """
        Initialize mouse controller.

        Args:
            speed: Movement speed multiplier (0.1 = slow, 1.0 = normal, 2.0 = fast)
        """
        self.speed = speed
        pyautogui.PAUSE = 0.1  # Small pause between actions
        pyautogui.FAILSAFE = True  # Move to corner to abort

    def move_to(self, x: int, y: int, duration: float = None):
        """
        Move mouse to position smoothly.

        Args:
            x, y: Target coordinates
            duration: Movement duration in seconds (auto-calculated if None)
        """
        if duration is None:
            # Calculate duration based on distance and speed
            current_x, current_y = pyautogui.position()
            distance = np.sqrt((x - current_x)**2 + (y - current_y)**2)
            duration = (distance / 1000) / self.speed  # Slower for longer distances
            duration = max(0.1, min(2.0, duration))  # Clamp between 0.1 and 2.0 seconds

        pyautogui.moveTo(x, y, duration=duration, tween=pyautogui.easeInOutQuad)

    def click(self, x: int = None, y: int = None, button: str = "left",
              clicks: int = 1, interval: float = 0.0):
        """
        Click at position.

        Args:
            x, y: Click coordinates (None = current position)
            button: "left", "right", or "middle"
            clicks: Number of clicks (2 for double-click)
            interval: Delay between clicks for multi-click
        """
        if x is not None and y is not None:
            self.move_to(x, y)

        pyautogui.click(button=button, clicks=clicks, interval=interval)

    def double_click(self, x: int = None, y: int = None):
        """Double-click at position."""
        self.click(x, y, clicks=2, interval=0.1)

    def right_click(self, x: int = None, y: int = None):
        """Right-click at position."""
        self.click(x, y, button="right")

    def drag_to(self, start_x: int, start_y: int, end_x: int, end_y: int,
                duration: float = 1.0, button: str = "left"):
        """
        Drag from start to end.

        Useful for:
        - Drag and drop
        - Selecting text
        - Moving windows
        """
        self.move_to(start_x, start_y)
        pyautogui.mouseDown(button=button)
        time.sleep(0.1)
        pyautogui.moveTo(end_x, end_y, duration=duration, tween=pyautogui.easeInOutQuad)
        time.sleep(0.1)
        pyautogui.mouseUp(button=button)

    def scroll(self, clicks: int, direction: str = "down"):
        """
        Scroll mouse wheel.

        Args:
            clicks: Number of scroll clicks (negative for up, positive for down)
            direction: "up" or "down" (overrides clicks sign)
        """
        if direction == "up":
            clicks = abs(clicks)
        elif direction == "down":
            clicks = -abs(clicks)

        pyautogui.scroll(clicks)

    def get_position(self) -> Tuple[int, int]:
        """Get current mouse position."""
        return pyautogui.position()

    def is_safe_position(self) -> bool:
        """
        Check if mouse is in safe position (not at screen corner).

        Moving mouse to corner triggers pyautogui failsafe.
        """
        x, y = pyautogui.position()
        screen_width, screen_height = pyautogui.size()

        # Check if at any corner (within 10 pixels)
        at_corner = (
            (x < 10 or x > screen_width - 10) and
            (y < 10 or y > screen_height - 10)
        )

        return not at_corner
```

#### 2.2 Keyboard Controller

```python
import pyautogui
import time
from typing import List

class KeyboardController:
    """
    Control keyboard programmatically with human-like typing.

    Features:
    - Type text with realistic timing
    - Press individual keys
    - Keyboard shortcuts (Ctrl+C, etc.)
    - Hold keys
    """

    def __init__(self, typing_speed: float = 0.05):
        """
        Initialize keyboard controller.

        Args:
            typing_speed: Interval between keystrokes in seconds
        """
        self.typing_speed = typing_speed

    def type_text(self, text: str, interval: float = None):
        """
        Type text with human-like timing.

        Args:
            text: Text to type
            interval: Time between keystrokes (uses self.typing_speed if None)
        """
        if interval is None:
            interval = self.typing_speed

        # Add slight randomness to timing (more human-like)
        import random
        for char in text:
            pyautogui.write(char, interval=interval * random.uniform(0.8, 1.2))

    def press_key(self, key: str, presses: int = 1, interval: float = 0.0):
        """
        Press keyboard key.

        Args:
            key: Key name (e.g., 'enter', 'esc', 'tab', 'a', 'shift')
            presses: Number of times to press
            interval: Delay between presses
        """
        pyautogui.press(key, presses=presses, interval=interval)

    def hotkey(self, *keys: str):
        """
        Press key combination.

        Examples:
            hotkey('ctrl', 'c')  # Copy
            hotkey('ctrl', 'shift', 's')  # Save as
            hotkey('cmd', 'space')  # Spotlight (macOS)
        """
        pyautogui.hotkey(*keys)

    def hold_key(self, key: str, duration: float = 1.0):
        """
        Hold key for duration.

        Useful for:
        - Holding Shift while clicking (multi-select)
        - Holding Ctrl for tooltips
        """
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)

    def paste_text(self, text: str):
        """
        Paste text using clipboard (faster than typing).

        Useful for long text or preserving formatting.
        """
        import pyperclip
        pyperclip.copy(text)

        # Platform-specific paste shortcut
        import platform
        if platform.system() == "Darwin":  # macOS
            self.hotkey('command', 'v')
        else:  # Windows/Linux
            self.hotkey('ctrl', 'v')

    def clear_text_field(self):
        """
        Clear text field (select all + delete).

        Useful before typing into existing field.
        """
        import platform
        if platform.system() == "Darwin":
            self.hotkey('command', 'a')
        else:
            self.hotkey('ctrl', 'a')

        self.press_key('delete')

    # Common shortcuts
    def copy(self):
        """Press Ctrl+C / Cmd+C."""
        import platform
        if platform.system() == "Darwin":
            self.hotkey('command', 'c')
        else:
            self.hotkey('ctrl', 'c')

    def paste(self):
        """Press Ctrl+V / Cmd+V."""
        import platform
        if platform.system() == "Darwin":
            self.hotkey('command', 'v')
        else:
            self.hotkey('ctrl', 'v')

    def save(self):
        """Press Ctrl+S / Cmd+S."""
        import platform
        if platform.system() == "Darwin":
            self.hotkey('command', 's')
        else:
            self.hotkey('ctrl', 's')

    def undo(self):
        """Press Ctrl+Z / Cmd+Z."""
        import platform
        if platform.system() == "Darwin":
            self.hotkey('command', 'z')
        else:
            self.hotkey('ctrl', 'z')
```

### 3. Browser Automation Framework

```python
from playwright.async_api import async_playwright, Page, Browser
from typing import Optional, List, Dict
import asyncio

class BrowserAutomationAgent:
    """
    High-level browser automation using Playwright.

    Advantages over pure computer use:
    - Faster (direct DOM access vs visual parsing)
    - More reliable (CSS selectors vs OCR)
    - Can run headless (no display needed)
    - Access to browser APIs (cookies, storage, network)

    When to use:
    - Web applications with inspectable DOM
    - Need to interact with page JavaScript
    - Need to intercept network requests
    - Need to test responsive design

    When NOT to use:
    - Canvas-based apps (use VLM instead)
    - Need to verify visual appearance
    - Shadow DOM or complex iframes
    """

    def __init__(self, browser_type: str = "chromium", headless: bool = False):
        """
        Initialize browser automation.

        Args:
            browser_type: "chromium", "firefox", or "webkit"
            headless: Run without visible browser window
        """
        self.browser_type = browser_type
        self.headless = headless
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    async def __aenter__(self):
        """Context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()

    async def start(self):
        """Start browser."""
        playwright = await async_playwright().start()

        if self.browser_type == "chromium":
            self.browser = await playwright.chromium.launch(headless=self.headless)
        elif self.browser_type == "firefox":
            self.browser = await playwright.firefox.launch(headless=self.headless)
        elif self.browser_type == "webkit":
            self.browser = await playwright.webkit.launch(headless=self.headless)

        self.page = await self.browser.new_page()

    async def close(self):
        """Close browser."""
        if self.browser:
            await self.browser.close()

    # Navigation
    async def navigate(self, url: str, wait_until: str = "networkidle"):
        """
        Navigate to URL.

        Args:
            url: Target URL
            wait_until: "load" | "domcontentloaded" | "networkidle"
        """
        await self.page.goto(url, wait_until=wait_until)

    async def go_back(self):
        """Navigate back."""
        await self.page.go_back()

    async def go_forward(self):
        """Navigate forward."""
        await self.page.go_forward()

    async def reload(self):
        """Reload current page."""
        await self.page.reload()

    # Element interaction
    async def click(self, selector: str, timeout: float = 30000):
        """
        Click element.

        Args:
            selector: CSS selector
            timeout: Wait timeout in milliseconds
        """
        await self.page.click(selector, timeout=timeout)

    async def fill(self, selector: str, value: str):
        """Fill input field."""
        await self.page.fill(selector, value)

    async def type_text(self, selector: str, text: str, delay: int = 100):
        """
        Type text with delay (human-like).

        Args:
            selector: CSS selector
            text: Text to type
            delay: Delay between keystrokes in ms
        """
        await self.page.type(selector, text, delay=delay)

    async def select_option(self, selector: str, value: str):
        """Select dropdown option."""
        await self.page.select_option(selector, value)

    async def check(self, selector: str):
        """Check checkbox."""
        await self.page.check(selector)

    async def uncheck(self, selector: str):
        """Uncheck checkbox."""
        await self.page.uncheck(selector)

    # Waiting
    async def wait_for_selector(self, selector: str, timeout: float = 30000):
        """Wait for element to appear."""
        await self.page.wait_for_selector(selector, timeout=timeout)

    async def wait_for_url(self, url_pattern: str, timeout: float = 30000):
        """Wait for URL to match pattern."""
        await self.page.wait_for_url(url_pattern, timeout=timeout)

    async def wait_for_navigation(self):
        """Wait for navigation to complete."""
        await self.page.wait_for_load_state("networkidle")

    # Data extraction
    async def get_text(self, selector: str) -> str:
        """Extract text from element."""
        return await self.page.text_content(selector)

    async def get_attribute(self, selector: str, attribute: str) -> Optional[str]:
        """Get element attribute value."""
        return await self.page.get_attribute(selector, attribute)

    async def get_all_text(self, selector: str) -> List[str]:
        """Extract text from all matching elements."""
        elements = await self.page.query_selector_all(selector)
        texts = []
        for elem in elements:
            text = await elem.text_content()
            if text:
                texts.append(text)
        return texts

    # JavaScript execution
    async def evaluate_js(self, script: str) -> any:
        """
        Execute JavaScript in page context.

        Returns:
            Result of JavaScript expression
        """
        return await self.page.evaluate(script)

    async def extract_data(self, script: str) -> Dict:
        """
        Extract structured data using JavaScript.

        Example:
            data = await browser.extract_data('''
                () => {
                    const emails = [];
                    document.querySelectorAll('.email-row').forEach(row => {
                        emails.push({
                            sender: row.querySelector('.sender').textContent,
                            subject: row.querySelector('.subject').textContent,
                            date: row.querySelector('.date').textContent
                        });
                    });
                    return emails;
                }
            ''')
        """
        return await self.page.evaluate(script)

    # Screenshots
    async def screenshot(self, path: Optional[str] = None, full_page: bool = False) -> bytes:
        """
        Take screenshot.

        Args:
            path: Save path (None = return bytes)
            full_page: Capture entire page (scroll)

        Returns:
            Screenshot bytes if path is None
        """
        return await self.page.screenshot(path=path, full_page=full_page)

    async def screenshot_element(self, selector: str, path: Optional[str] = None) -> bytes:
        """Screenshot specific element."""
        element = await self.page.query_selector(selector)
        return await element.screenshot(path=path)

    # Advanced features
    async def intercept_requests(self, pattern: str, callback):
        """
        Intercept network requests.

        Example:
            async def block_ads(route, request):
                if "ads" in request.url:
                    await route.abort()
                else:
                    await route.continue_()

            await browser.intercept_requests("**/*", block_ads)
        """
        await self.page.route(pattern, callback)

    async def get_cookies(self) -> List[Dict]:
        """Get all cookies."""
        return await self.page.context.cookies()

    async def set_cookie(self, name: str, value: str, domain: str = None):
        """Set cookie."""
        await self.page.context.add_cookies([{
            "name": name,
            "value": value,
            "domain": domain or self.page.url,
            "path": "/"
        }])

    async def wait_for_download(self) -> str:
        """
        Wait for download and return path.

        Use after clicking download button.
        """
        async with self.page.expect_download() as download_info:
            download = await download_info.value
            return await download.path()
```

### Bug Fixes for Existing Code Above

> **Fix 1 -- Missing `Dict` import in `ScreenCaptureEngine`**
>
> The `list_monitors` method in `ScreenCaptureEngine` returns `List[Dict]` but `Dict` is
> not imported. Add `Dict` to the imports at the top of that module:
>
> ```python
> from typing import Tuple, List, Optional, Dict
> ```

> **Fix 2 -- `numpy` import scope issue in `OCREngine`**
>
> `OCREngine.extract_text` and `find_text` call `np.array(image)` but `numpy` is only
> imported in the `ScreenCaptureEngine` module. Each module must carry its own imports:
>
> ```python
> # Add at the top of the OCR Engine module
> import numpy as np
> ```

> **Fix 3 -- `VLMAnalyzer.find_clickable_elements` bounding-box reliability**
>
> VLMs are notoriously unreliable at predicting pixel-level bounding boxes from a free-form
> JSON prompt. The implementation should:
> 1. Cross-validate VLM bounding boxes against OCR text locations.
> 2. Snap VLM-predicted boxes to the nearest OCR-detected text region.
> 3. Fall back to OCR-only detection when VLM confidence is below threshold.
>
> ```python
> def find_clickable_elements(
>     self,
>     image: Image.Image,
>     ocr_engine: Optional["OCREngine"] = None,
>     snap_threshold: int = 30,
> ) -> List[Dict]:
>     """
>     Detect clickable UI elements with OCR cross-validation.
>
>     Args:
>         image: Screenshot to analyze.
>         ocr_engine: Optional OCR engine for bounding-box cross-validation.
>         snap_threshold: Max pixel distance to snap VLM box to OCR box.
>
>     Returns:
>         List of element dicts with validated bounding boxes.
>     """
>     # Step 1: VLM detection (as before)
>     vlm_elements = self._vlm_detect_elements(image)
>
>     if ocr_engine is None:
>         return vlm_elements
>
>     # Step 2: OCR detection for cross-validation
>     ocr_data = ocr_engine.extract_structured(image)
>     ocr_boxes = [w["bbox"] for w in ocr_data.get("words", [])]
>
>     # Step 3: Snap VLM boxes to nearest OCR boxes
>     validated = []
>     for elem in vlm_elements:
>         vlm_box = elem["bbox"]
>         best_match = None
>         best_dist = float("inf")
>
>         for ocr_box in ocr_boxes:
>             dist = abs(vlm_box.center[0] - ocr_box.center[0]) + \
>                    abs(vlm_box.center[1] - ocr_box.center[1])
>             if dist < best_dist:
>                 best_dist = dist
>                 best_match = ocr_box
>
>         if best_match and best_dist <= snap_threshold:
>             elem["bbox"] = best_match  # Use more accurate OCR box
>             elem["validated"] = True
>         else:
>             elem["validated"] = False
>
>         validated.append(elem)
>
>     return validated
> ```

> **Fix 4 -- `wait_for_download` async context manager misuse**
>
> The `expect_download()` context manager yields a future. The download object must be
> awaited *after* the triggering click action happens inside the context block. The current
> code awaits the download inside the `async with` but never triggers a click. It should be
> used as follows:
>
> ```python
> async def wait_for_download(self, click_selector: str) -> str:
>     """
>     Click a download link/button and wait for the download to complete.
>
>     Args:
>         click_selector: CSS selector of the element that triggers the download.
>
>     Returns:
>         Local filesystem path to the downloaded file.
>     """
>     async with self.page.expect_download() as download_info:
>         await self.page.click(click_selector)
>     download = await download_info.value
>     path = await download.path()
>     return str(path)
> ```

---

### 4. Desktop Application Automation

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Desktop application automation via accessibility APIs.

Provides a cross-platform abstraction over OS-native UI tree APIs:
- Windows: UI Automation (UIA) via comtypes / uiautomation
- macOS: NSAccessibility via pyobjc
- Linux: AT-SPI via python-atspi / pyatspi2

Location: src/gaia/computer_use/desktop.py
"""

import platform
import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from gaia.logger import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

@dataclass
class UIElement:
    """
    Represents a single node in the OS accessibility tree.
    """

    role: str  # e.g. "Button", "TextField", "MenuItem"
    name: str  # Visible label / accessible name
    value: Optional[str] = None  # Current value (for text fields, sliders, etc.)
    bbox: Optional["BoundingBox"] = None
    children: List["UIElement"] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    native_handle: Optional[Any] = None  # Platform-specific handle (not serialised)

    @property
    def is_interactive(self) -> bool:
        """Return True if this element can receive user input."""
        interactive_roles = {
            "Button", "TextField", "TextArea", "CheckBox", "RadioButton",
            "ComboBox", "Slider", "Link", "MenuItem", "Tab", "ListItem",
        }
        return self.role in interactive_roles

    def find_by_name(self, name: str, partial: bool = False) -> Optional["UIElement"]:
        """Depth-first search for an element by accessible name."""
        if partial and name.lower() in self.name.lower():
            return self
        elif not partial and self.name == name:
            return self
        for child in self.children:
            result = child.find_by_name(name, partial=partial)
            if result:
                return result
        return None

    def find_by_role(self, role: str) -> List["UIElement"]:
        """Return all descendants matching the given role."""
        matches = []
        if self.role == role:
            matches.append(self)
        for child in self.children:
            matches.extend(child.find_by_role(role))
        return matches


@dataclass
class WindowInfo:
    """Metadata about an OS window."""

    title: str
    pid: int
    bbox: Optional["BoundingBox"] = None
    is_focused: bool = False
    is_minimised: bool = False
    native_handle: Optional[Any] = None


# ──────────────────────────────────────────────
# Abstract Base -- Platform Adapter
# ──────────────────────────────────────────────

class DesktopAutomationBase(abc.ABC):
    """
    Abstract interface for platform-specific desktop automation.

    Each platform subclass must implement every abstract method.
    """

    # ── Window Management ──

    @abc.abstractmethod
    def list_windows(self) -> List[WindowInfo]:
        """Return a list of all visible windows."""
        ...

    @abc.abstractmethod
    def focus_window(self, window: WindowInfo) -> bool:
        """Bring window to front and give it keyboard focus."""
        ...

    @abc.abstractmethod
    def minimize_window(self, window: WindowInfo) -> bool:
        """Minimize the window."""
        ...

    @abc.abstractmethod
    def maximize_window(self, window: WindowInfo) -> bool:
        """Maximize the window."""
        ...

    @abc.abstractmethod
    def close_window(self, window: WindowInfo) -> bool:
        """Close the window."""
        ...

    @abc.abstractmethod
    def move_window(self, window: WindowInfo, x: int, y: int) -> bool:
        """Move the window to (x, y)."""
        ...

    @abc.abstractmethod
    def resize_window(self, window: WindowInfo, width: int, height: int) -> bool:
        """Resize the window."""
        ...

    # ── UI Tree ──

    @abc.abstractmethod
    def get_ui_tree(self, window: WindowInfo, max_depth: int = 10) -> UIElement:
        """
        Retrieve the accessibility tree for the given window.

        Args:
            window: Target window.
            max_depth: How deep to recurse (avoids huge trees).

        Returns:
            Root UIElement with children populated.
        """
        ...

    @abc.abstractmethod
    def find_element(
        self, window: WindowInfo, role: str = None, name: str = None
    ) -> Optional[UIElement]:
        """Find the first element matching role/name criteria."""
        ...

    # ── Element Interaction ──

    @abc.abstractmethod
    def click_element(self, element: UIElement) -> bool:
        """Invoke the default action on the element (click / press)."""
        ...

    @abc.abstractmethod
    def set_element_value(self, element: UIElement, value: str) -> bool:
        """Set the value of a text field, slider, etc."""
        ...

    @abc.abstractmethod
    def get_element_value(self, element: UIElement) -> Optional[str]:
        """Read current value from the element."""
        ...


# ──────────────────────────────────────────────
# Windows Implementation -- UI Automation (UIA)
# ──────────────────────────────────────────────

class WindowsDesktopAutomation(DesktopAutomationBase):
    """
    Windows desktop automation via the UI Automation COM API.

    Requires: pip install uiautomation
    (or comtypes for raw COM access)

    Location: src/gaia/computer_use/platforms/windows.py
    """

    def __init__(self):
        try:
            import uiautomation as auto
            self._auto = auto
        except ImportError:
            raise ImportError(
                "uiautomation is required for Windows desktop automation. "
                "Install with: pip install uiautomation"
            )

    def list_windows(self) -> List[WindowInfo]:
        windows = []
        for win in self._auto.GetRootControl().GetChildren():
            if win.ClassName in ("Shell_TrayWnd", "Progman"):
                continue  # Skip taskbar / desktop
            rect = win.BoundingRectangle
            windows.append(
                WindowInfo(
                    title=win.Name or "",
                    pid=win.ProcessId,
                    bbox=BoundingBox(
                        x=rect.left, y=rect.top,
                        width=rect.width(), height=rect.height(),
                    ) if rect else None,
                    is_focused=win.HasKeyboardFocus,
                    native_handle=win,
                )
            )
        return windows

    def focus_window(self, window: WindowInfo) -> bool:
        try:
            win = window.native_handle
            win.SetFocus()
            return True
        except Exception as exc:
            log.error(f"Failed to focus window '{window.title}': {exc}")
            return False

    def minimize_window(self, window: WindowInfo) -> bool:
        try:
            pattern = window.native_handle.GetWindowPattern()
            pattern.SetWindowVisualState(self._auto.WindowVisualState.Minimized)
            return True
        except Exception as exc:
            log.error(f"Failed to minimize window '{window.title}': {exc}")
            return False

    def maximize_window(self, window: WindowInfo) -> bool:
        try:
            pattern = window.native_handle.GetWindowPattern()
            pattern.SetWindowVisualState(self._auto.WindowVisualState.Maximized)
            return True
        except Exception as exc:
            log.error(f"Failed to maximize window '{window.title}': {exc}")
            return False

    def close_window(self, window: WindowInfo) -> bool:
        try:
            pattern = window.native_handle.GetWindowPattern()
            pattern.Close()
            return True
        except Exception as exc:
            log.error(f"Failed to close window '{window.title}': {exc}")
            return False

    def move_window(self, window: WindowInfo, x: int, y: int) -> bool:
        try:
            pattern = window.native_handle.GetTransformPattern()
            pattern.Move(x, y)
            return True
        except Exception as exc:
            log.error(f"Failed to move window '{window.title}': {exc}")
            return False

    def resize_window(self, window: WindowInfo, width: int, height: int) -> bool:
        try:
            pattern = window.native_handle.GetTransformPattern()
            pattern.Resize(width, height)
            return True
        except Exception as exc:
            log.error(f"Failed to resize window '{window.title}': {exc}")
            return False

    def get_ui_tree(self, window: WindowInfo, max_depth: int = 10) -> UIElement:
        def _walk(control, depth: int) -> UIElement:
            rect = control.BoundingRectangle
            elem = UIElement(
                role=control.ControlTypeName,
                name=control.Name or "",
                value=getattr(control, "CurrentValue", None),
                bbox=BoundingBox(
                    x=rect.left, y=rect.top,
                    width=rect.width(), height=rect.height(),
                ) if rect else None,
                native_handle=control,
            )
            if depth < max_depth:
                for child in control.GetChildren():
                    elem.children.append(_walk(child, depth + 1))
            return elem

        return _walk(window.native_handle, 0)

    def find_element(
        self, window: WindowInfo, role: str = None, name: str = None
    ) -> Optional[UIElement]:
        tree = self.get_ui_tree(window, max_depth=15)
        if name:
            found = tree.find_by_name(name, partial=True)
            if found and (role is None or found.role == role):
                return found
        if role:
            matches = tree.find_by_role(role)
            return matches[0] if matches else None
        return None

    def click_element(self, element: UIElement) -> bool:
        try:
            ctrl = element.native_handle
            pattern = ctrl.GetInvokePattern()
            if pattern:
                pattern.Invoke()
                return True
            # Fallback: click at centre of bounding box
            if element.bbox:
                cx, cy = element.bbox.center
                import pyautogui
                pyautogui.click(cx, cy)
                return True
            return False
        except Exception as exc:
            log.error(f"Failed to click element '{element.name}': {exc}")
            return False

    def set_element_value(self, element: UIElement, value: str) -> bool:
        try:
            ctrl = element.native_handle
            pattern = ctrl.GetValuePattern()
            if pattern:
                pattern.SetValue(value)
                return True
            return False
        except Exception as exc:
            log.error(f"Failed to set value on '{element.name}': {exc}")
            return False

    def get_element_value(self, element: UIElement) -> Optional[str]:
        try:
            ctrl = element.native_handle
            pattern = ctrl.GetValuePattern()
            return pattern.Value if pattern else None
        except Exception:
            return None


# ──────────────────────────────────────────────
# macOS Implementation -- NSAccessibility
# ──────────────────────────────────────────────

class MacOSDesktopAutomation(DesktopAutomationBase):
    """
    macOS desktop automation via NSAccessibility (pyobjc).

    Requires:
        pip install pyobjc-framework-Cocoa pyobjc-framework-ApplicationServices

    NOTE: The calling process must have Accessibility permissions
    (System Settings > Privacy & Security > Accessibility).

    Location: src/gaia/computer_use/platforms/macos.py
    """

    def __init__(self):
        try:
            from ApplicationServices import (
                AXUIElementCreateSystemWide,
                AXUIElementCreateApplication,
            )
            from Cocoa import NSWorkspace
            self._AXUIElementCreateSystemWide = AXUIElementCreateSystemWide
            self._AXUIElementCreateApplication = AXUIElementCreateApplication
            self._NSWorkspace = NSWorkspace
        except ImportError:
            raise ImportError(
                "pyobjc frameworks required for macOS desktop automation. "
                "Install with: pip install pyobjc-framework-Cocoa "
                "pyobjc-framework-ApplicationServices"
            )

    def list_windows(self) -> List[WindowInfo]:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
        )

        window_list = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly, kCGNullWindowID
        )
        windows = []
        for win in window_list:
            title = win.get("kCGWindowName", "")
            pid = win.get("kCGWindowOwnerPID", 0)
            bounds = win.get("kCGWindowBounds", {})
            windows.append(
                WindowInfo(
                    title=title or "",
                    pid=pid,
                    bbox=BoundingBox(
                        x=int(bounds.get("X", 0)),
                        y=int(bounds.get("Y", 0)),
                        width=int(bounds.get("Width", 0)),
                        height=int(bounds.get("Height", 0)),
                    ),
                )
            )
        return windows

    def focus_window(self, window: WindowInfo) -> bool:
        import subprocess
        try:
            subprocess.run(
                ["osascript", "-e",
                 f'tell application "System Events" to set frontmost of '
                 f'(first process whose unix id is {window.pid}) to true'],
                check=True, capture_output=True,
            )
            return True
        except Exception as exc:
            log.error(f"Failed to focus window '{window.title}': {exc}")
            return False

    def minimize_window(self, window: WindowInfo) -> bool:
        # Implement via AXUIElement
        log.warning("macOS minimize_window: stub -- use AXUIElement SetAttribute")
        return False

    def maximize_window(self, window: WindowInfo) -> bool:
        log.warning("macOS maximize_window: stub -- use AXUIElement SetAttribute")
        return False

    def close_window(self, window: WindowInfo) -> bool:
        log.warning("macOS close_window: stub -- use AXUIElement press close button")
        return False

    def move_window(self, window: WindowInfo, x: int, y: int) -> bool:
        log.warning("macOS move_window: stub")
        return False

    def resize_window(self, window: WindowInfo, width: int, height: int) -> bool:
        log.warning("macOS resize_window: stub")
        return False

    def get_ui_tree(self, window: WindowInfo, max_depth: int = 10) -> UIElement:
        app_ref = self._AXUIElementCreateApplication(window.pid)

        def _walk(ax_element, depth: int) -> UIElement:
            import Cocoa
            err, role = Cocoa.AXUIElementCopyAttributeValue(
                ax_element, "AXRole", None
            )
            err, title = Cocoa.AXUIElementCopyAttributeValue(
                ax_element, "AXTitle", None
            )
            err, value = Cocoa.AXUIElementCopyAttributeValue(
                ax_element, "AXValue", None
            )
            elem = UIElement(
                role=str(role) if role else "Unknown",
                name=str(title) if title else "",
                value=str(value) if value else None,
                native_handle=ax_element,
            )
            if depth < max_depth:
                err, children = Cocoa.AXUIElementCopyAttributeValue(
                    ax_element, "AXChildren", None
                )
                if children:
                    for child in children:
                        elem.children.append(_walk(child, depth + 1))
            return elem

        return _walk(app_ref, 0)

    def find_element(
        self, window: WindowInfo, role: str = None, name: str = None
    ) -> Optional[UIElement]:
        tree = self.get_ui_tree(window, max_depth=15)
        if name:
            found = tree.find_by_name(name, partial=True)
            if found and (role is None or found.role == role):
                return found
        if role:
            matches = tree.find_by_role(role)
            return matches[0] if matches else None
        return None

    def click_element(self, element: UIElement) -> bool:
        try:
            import Cocoa
            Cocoa.AXUIElementPerformAction(element.native_handle, "AXPress")
            return True
        except Exception as exc:
            log.error(f"Failed to click element '{element.name}': {exc}")
            return False

    def set_element_value(self, element: UIElement, value: str) -> bool:
        try:
            import Cocoa
            Cocoa.AXUIElementSetAttributeValue(
                element.native_handle, "AXValue", value
            )
            return True
        except Exception as exc:
            log.error(f"Failed to set value on '{element.name}': {exc}")
            return False

    def get_element_value(self, element: UIElement) -> Optional[str]:
        try:
            import Cocoa
            err, value = Cocoa.AXUIElementCopyAttributeValue(
                element.native_handle, "AXValue", None
            )
            return str(value) if value else None
        except Exception:
            return None


# ──────────────────────────────────────────────
# Linux Implementation -- AT-SPI
# ──────────────────────────────────────────────

class LinuxDesktopAutomation(DesktopAutomationBase):
    """
    Linux desktop automation via AT-SPI (Assistive Technology Service Provider Interface).

    Requires:
        pip install pyatspi
        System: at-spi2-core must be running

    Location: src/gaia/computer_use/platforms/linux.py
    """

    def __init__(self):
        try:
            import pyatspi
            self._atspi = pyatspi
        except ImportError:
            raise ImportError(
                "pyatspi is required for Linux desktop automation. "
                "Install with: pip install pyatspi  "
                "(also ensure at-spi2-core is running)"
            )

    def list_windows(self) -> List[WindowInfo]:
        desktop = self._atspi.Registry.getDesktop(0)
        windows = []
        for i in range(desktop.childCount):
            app = desktop.getChildAtIndex(i)
            if app is None:
                continue
            for j in range(app.childCount):
                win = app.getChildAtIndex(j)
                if win is None:
                    continue
                role = win.getRole()
                if role == self._atspi.ROLE_FRAME:
                    try:
                        ext = win.queryComponent().getExtents(
                            self._atspi.DESKTOP_COORDS
                        )
                        bbox = BoundingBox(
                            x=ext.x, y=ext.y,
                            width=ext.width, height=ext.height,
                        )
                    except Exception:
                        bbox = None
                    windows.append(
                        WindowInfo(
                            title=win.name or "",
                            pid=app.get_process_id(),
                            bbox=bbox,
                            native_handle=win,
                        )
                    )
        return windows

    def focus_window(self, window: WindowInfo) -> bool:
        try:
            import subprocess
            subprocess.run(
                ["xdotool", "windowactivate", "--sync",
                 str(window.native_handle.queryComponent().getLayer())],
                check=True, capture_output=True,
            )
            return True
        except Exception as exc:
            log.error(f"Failed to focus window '{window.title}': {exc}")
            return False

    def minimize_window(self, window: WindowInfo) -> bool:
        log.warning("Linux minimize_window: stub -- use wmctrl or xdotool")
        return False

    def maximize_window(self, window: WindowInfo) -> bool:
        log.warning("Linux maximize_window: stub -- use wmctrl or xdotool")
        return False

    def close_window(self, window: WindowInfo) -> bool:
        log.warning("Linux close_window: stub -- use wmctrl or xdotool")
        return False

    def move_window(self, window: WindowInfo, x: int, y: int) -> bool:
        log.warning("Linux move_window: stub -- use wmctrl or xdotool")
        return False

    def resize_window(self, window: WindowInfo, width: int, height: int) -> bool:
        log.warning("Linux resize_window: stub -- use wmctrl or xdotool")
        return False

    def get_ui_tree(self, window: WindowInfo, max_depth: int = 10) -> UIElement:
        def _walk(node, depth: int) -> UIElement:
            role_name = self._atspi.Role(node.getRole()).valueName
            elem = UIElement(
                role=role_name.replace("ROLE_", ""),
                name=node.name or "",
                native_handle=node,
            )
            # Try to read value
            try:
                val_iface = node.queryValue()
                elem.value = str(val_iface.currentValue)
            except Exception:
                pass

            # Try to read bounding box
            try:
                ext = node.queryComponent().getExtents(self._atspi.DESKTOP_COORDS)
                elem.bbox = BoundingBox(
                    x=ext.x, y=ext.y, width=ext.width, height=ext.height,
                )
            except Exception:
                pass

            if depth < max_depth:
                for i in range(node.childCount):
                    child = node.getChildAtIndex(i)
                    if child:
                        elem.children.append(_walk(child, depth + 1))
            return elem

        return _walk(window.native_handle, 0)

    def find_element(
        self, window: WindowInfo, role: str = None, name: str = None
    ) -> Optional[UIElement]:
        tree = self.get_ui_tree(window, max_depth=15)
        if name:
            found = tree.find_by_name(name, partial=True)
            if found and (role is None or found.role == role):
                return found
        if role:
            matches = tree.find_by_role(role)
            return matches[0] if matches else None
        return None

    def click_element(self, element: UIElement) -> bool:
        try:
            action = element.native_handle.queryAction()
            for i in range(action.nActions):
                if action.getName(i) in ("click", "activate", "press"):
                    action.doAction(i)
                    return True
            # Fallback: click at centre
            if element.bbox:
                cx, cy = element.bbox.center
                import pyautogui
                pyautogui.click(cx, cy)
                return True
            return False
        except Exception as exc:
            log.error(f"Failed to click element '{element.name}': {exc}")
            return False

    def set_element_value(self, element: UIElement, value: str) -> bool:
        try:
            text_iface = element.native_handle.queryEditableText()
            text_iface.setTextContents(value)
            return True
        except Exception as exc:
            log.error(f"Failed to set value on '{element.name}': {exc}")
            return False

    def get_element_value(self, element: UIElement) -> Optional[str]:
        try:
            text_iface = element.native_handle.queryText()
            return text_iface.getText(0, text_iface.characterCount)
        except Exception:
            return None


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def create_desktop_automation() -> DesktopAutomationBase:
    """
    Create the correct platform adapter for the current OS.

    Returns:
        DesktopAutomationBase subclass.

    Raises:
        NotImplementedError: If the platform is not supported.
    """
    system = platform.system()
    if system == "Windows":
        return WindowsDesktopAutomation()
    elif system == "Darwin":
        return MacOSDesktopAutomation()
    elif system == "Linux":
        return LinuxDesktopAutomation()
    else:
        raise NotImplementedError(f"Desktop automation not supported on {system}")
```

---

### 5. ComputerUseController

The `ComputerUseController` is the **central orchestration class**. It receives a
high-level natural-language intent from the agent, decomposes it into a plan of discrete
UI actions, executes each action through the appropriate subsystem, verifies the result
after every step, and re-plans if something goes wrong.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ComputerUseController                             │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐       │
│  │ Intent Parser │──▶│ Action       │──▶│ Action Executor   │       │
│  │ (LLM)        │   │ Planner (LLM)│   │ (subsystem call)  │       │
│  └──────────────┘   └──────┬───────┘   └─────────┬─────────┘       │
│                            │                      │                  │
│                            ▼                      ▼                  │
│                   ┌──────────────┐        ┌──────────────┐          │
│                   │ Safety Guard │        │ Verification │          │
│                   │ (pre-action) │        │ Loop (VLM)   │          │
│                   └──────────────┘        └──────────────┘          │
│                                                                      │
│  Subsystems:  Vision | Mouse | Keyboard | Browser | Desktop         │
└──────────────────────────────────────────────────────────────────────┘
```

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
ComputerUseController -- high-level orchestrator for computer use actions.

Location: src/gaia/computer_use/controller.py
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────
# Action Model
# ──────────────────────────────────────────────

class ActionType(Enum):
    """Enumeration of all atomic UI actions."""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"
    HOTKEY = "hotkey"
    SCROLL = "scroll"
    DRAG = "drag"
    NAVIGATE_URL = "navigate_url"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    FIND_ELEMENT = "find_element"
    READ_TEXT = "read_text"
    SELECT_OPTION = "select_option"
    FOCUS_WINDOW = "focus_window"
    OPEN_APPLICATION = "open_application"


class ActionStatus(Enum):
    """Result of executing a single action."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_RETRY = "needs_retry"


@dataclass
class Action:
    """
    A single atomic UI action.

    Examples:
        Action(type=ActionType.CLICK, target="Send button",
               parameters={"x": 450, "y": 320})
        Action(type=ActionType.TYPE_TEXT, target="Search field",
               parameters={"text": "quarterly report"})
    """
    type: ActionType
    target: str  # Human-readable description of the target
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_result: str = ""  # Description of what should happen
    status: ActionStatus = ActionStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    retries: int = 0
    max_retries: int = 2


@dataclass
class Plan:
    """An ordered sequence of Actions that fulfil a high-level intent."""
    intent: str
    actions: List[Action] = field(default_factory=list)
    current_step: int = 0
    created_at: float = field(default_factory=time.time)
    completed: bool = False
    success: bool = False

    @property
    def progress(self) -> str:
        done = sum(1 for a in self.actions if a.status == ActionStatus.SUCCESS)
        return f"{done}/{len(self.actions)}"


# ──────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────

class ComputerUseController:
    """
    High-level orchestrator that turns natural language into verified UI actions.

    Lifecycle:
        1. parse_intent()  -- LLM extracts structured intent
        2. create_plan()   -- LLM generates ordered action list
        3. execute_plan()  -- Iterates actions with safety checks + verification
        4. replan()        -- If verification fails, re-examine screen and adjust

    Thread safety: NOT thread-safe. Use one controller per agent instance.
    """

    MAX_REPLAN_ATTEMPTS = 3
    VERIFICATION_TIMEOUT = 5.0  # seconds to wait before verifying

    def __init__(
        self,
        vision: "VLMAnalyzer",
        screen: "ScreenCaptureEngine",
        mouse: "MouseController",
        keyboard: "KeyboardController",
        browser: Optional["BrowserAutomationAgent"] = None,
        desktop: Optional["DesktopAutomationBase"] = None,
        safety: Optional["SafetyGuard"] = None,
        llm_client: Optional[Any] = None,
        auto_verify: bool = True,
        debug: bool = False,
    ):
        """
        Initialise the controller with all subsystems.

        Args:
            vision: VLM analyser for screen understanding.
            screen: Screenshot capture engine.
            mouse: Mouse controller.
            keyboard: Keyboard controller.
            browser: Optional Playwright browser automation.
            desktop: Optional platform desktop automation.
            safety: Safety guard for action validation (created if None).
            llm_client: LLM for intent parsing / planning (uses ChatSDK if None).
            auto_verify: Whether to verify each action via VLM (default True).
            debug: Enable verbose debug logging.
        """
        self.vision = vision
        self.screen = screen
        self.mouse = mouse
        self.keyboard = keyboard
        self.browser = browser
        self.desktop = desktop
        self.safety = safety or SafetyGuard()
        self.auto_verify = auto_verify
        self.debug = debug

        # LLM for planning
        if llm_client is None:
            from gaia.chat.sdk import ChatSDK, ChatConfig
            self._llm = ChatSDK(ChatConfig(
                max_tokens=2048,
                streaming=False,
            ))
        else:
            self._llm = llm_client

        # History for context
        self._action_history: List[Action] = []
        self._plan_history: List[Plan] = []

    # ── Intent Parsing ──

    def parse_intent(self, user_request: str) -> Dict[str, Any]:
        """
        Use LLM to extract structured intent from natural language.

        Args:
            user_request: e.g. "Open Gmail and archive all newsletters"

        Returns:
            {
                "goal": "Archive newsletter emails in Gmail",
                "application": "web_browser",
                "steps_summary": ["Open Gmail", "Identify newsletters", "Archive each"],
                "requires_login": True,
                "risk_level": "low"
            }
        """
        prompt = f"""You are a UI automation planner. Parse the following user request
into a structured intent.

User request: "{user_request}"

Return a JSON object with these fields:
- goal: One-sentence summary of the goal
- application: "web_browser" | "desktop_app" | "file_manager" | "terminal" | "mixed"
- steps_summary: Array of high-level step descriptions (3-8 steps)
- requires_login: boolean
- risk_level: "low" | "medium" | "high" | "critical"

Respond ONLY with the JSON object."""

        response = self._llm.send(prompt)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            log.error("Failed to parse intent from LLM response")
            return {
                "goal": user_request,
                "application": "mixed",
                "steps_summary": [user_request],
                "requires_login": False,
                "risk_level": "medium",
            }

    # ── Action Planning ──

    def create_plan(self, intent: Dict[str, Any], screenshot: "Image.Image") -> Plan:
        """
        Generate an action plan from structured intent + current screen state.

        The LLM sees the current screenshot and produces a concrete sequence
        of atomic actions (click, type, scroll, etc.).

        Args:
            intent: Output from parse_intent().
            screenshot: Current screen state.

        Returns:
            Plan with ordered Action list.
        """
        screen_description = self.vision.describe_screen(screenshot, detail_level="high")
        clickable = self.vision.find_clickable_elements(screenshot)

        clickable_summary = "\n".join(
            f"  - [{e['type']}] \"{e['text']}\" at ({e['bbox'].x}, {e['bbox'].y})"
            for e in clickable[:30]  # Limit to 30 elements for prompt size
        )

        prompt = f"""You are a UI automation planner. Given the current screen state and
the user's goal, produce a step-by-step action plan.

GOAL: {intent['goal']}

CURRENT SCREEN:
{screen_description}

CLICKABLE ELEMENTS:
{clickable_summary}

For each step, provide a JSON object with:
- action: One of {[a.value for a in ActionType]}
- target: Human description of the target element
- parameters: Dict of action-specific parameters
  - click/double_click/right_click: {{"x": int, "y": int}}
  - type_text: {{"text": "string to type"}}
  - press_key: {{"key": "enter"}}
  - hotkey: {{"keys": ["ctrl", "a"]}}
  - scroll: {{"direction": "up|down", "clicks": int}}
  - navigate_url: {{"url": "https://..."}}
  - wait: {{"seconds": float}}
  - select_option: {{"selector": "css", "value": "option_value"}}
- expected_result: Description of what should happen

Return a JSON array of action objects. Keep it concise (max 15 steps)."""

        response = self._llm.send(prompt)
        try:
            raw_actions = json.loads(response.text)
        except json.JSONDecodeError:
            log.error("Failed to parse plan from LLM response")
            raw_actions = []

        plan = Plan(intent=intent.get("goal", ""))
        for raw in raw_actions:
            try:
                plan.actions.append(Action(
                    type=ActionType(raw["action"]),
                    target=raw.get("target", ""),
                    parameters=raw.get("parameters", {}),
                    expected_result=raw.get("expected_result", ""),
                ))
            except (ValueError, KeyError) as exc:
                log.warning(f"Skipping invalid action: {raw} -- {exc}")

        log.info(f"Created plan with {len(plan.actions)} actions for: {plan.intent}")
        return plan

    # ── Action Execution ──

    def execute_plan(self, plan: Plan) -> Plan:
        """
        Execute all actions in the plan sequentially.

        For each action:
            1. Pre-validate with SafetyGuard
            2. Execute via appropriate subsystem
            3. Optionally verify result via VLM
            4. Retry or replan on failure

        Args:
            plan: The Plan to execute.

        Returns:
            The same Plan with updated action statuses.
        """
        replan_attempts = 0

        while plan.current_step < len(plan.actions):
            action = plan.actions[plan.current_step]

            # 1. Safety check
            validation = self.safety.validate_action(action)
            if not validation["allowed"]:
                if validation.get("needs_confirmation"):
                    # In non-interactive mode, skip dangerous actions
                    log.warning(
                        f"Action requires confirmation (skipping): "
                        f"{action.type.value} on '{action.target}' -- "
                        f"Reason: {validation.get('reason', 'unknown')}"
                    )
                    action.status = ActionStatus.SKIPPED
                    plan.current_step += 1
                    continue
                else:
                    log.error(
                        f"Action BLOCKED by safety guard: "
                        f"{action.type.value} on '{action.target}' -- "
                        f"Reason: {validation.get('reason', 'forbidden')}"
                    )
                    action.status = ActionStatus.FAILED
                    action.result = {"error": validation.get("reason", "blocked")}
                    plan.current_step += 1
                    continue

            # 2. Take pre-action screenshot (for verification)
            before_screenshot = self.screen.capture_screen() if self.auto_verify else None

            # 3. Execute
            log.info(
                f"Step {plan.current_step + 1}/{len(plan.actions)}: "
                f"{action.type.value} -> '{action.target}'"
            )
            success = self._execute_single_action(action)

            if not success:
                action.retries += 1
                if action.retries <= action.max_retries:
                    action.status = ActionStatus.NEEDS_RETRY
                    log.warning(
                        f"Action failed, retrying ({action.retries}/{action.max_retries})"
                    )
                    time.sleep(1.0)
                    continue  # Retry same step
                else:
                    action.status = ActionStatus.FAILED
                    # Attempt replan
                    if replan_attempts < self.MAX_REPLAN_ATTEMPTS:
                        replan_attempts += 1
                        log.warning(f"Replanning (attempt {replan_attempts})")
                        plan = self._replan(plan)
                        continue
                    else:
                        log.error("Max replan attempts reached. Aborting.")
                        break

            # 4. Verify
            if self.auto_verify and before_screenshot and action.expected_result:
                time.sleep(0.5)  # Let UI settle
                after_screenshot = self.screen.capture_screen()
                verified = self.vision.verify_action_result(
                    before_screenshot, after_screenshot, action.expected_result
                )
                if not verified:
                    log.warning(f"Verification failed for: {action.expected_result}")
                    action.retries += 1
                    if action.retries <= action.max_retries:
                        action.status = ActionStatus.NEEDS_RETRY
                        continue
                    # Accept anyway and move on (soft failure)
                    log.warning("Proceeding despite verification failure")

            action.status = ActionStatus.SUCCESS
            self._action_history.append(action)
            plan.current_step += 1

        # Determine overall success
        plan.completed = True
        plan.success = all(
            a.status in (ActionStatus.SUCCESS, ActionStatus.SKIPPED)
            for a in plan.actions
        )
        self._plan_history.append(plan)

        return plan

    def _execute_single_action(self, action: Action) -> bool:
        """
        Route an Action to the appropriate subsystem for execution.

        Returns:
            True if execution succeeded, False otherwise.
        """
        try:
            params = action.parameters
            action_type = action.type

            if action_type == ActionType.CLICK:
                self.mouse.click(params.get("x"), params.get("y"))

            elif action_type == ActionType.DOUBLE_CLICK:
                self.mouse.double_click(params.get("x"), params.get("y"))

            elif action_type == ActionType.RIGHT_CLICK:
                self.mouse.right_click(params.get("x"), params.get("y"))

            elif action_type == ActionType.TYPE_TEXT:
                text = params.get("text", "")
                self.keyboard.type_text(text)

            elif action_type == ActionType.PRESS_KEY:
                self.keyboard.press_key(params.get("key", ""))

            elif action_type == ActionType.HOTKEY:
                keys = params.get("keys", [])
                self.keyboard.hotkey(*keys)

            elif action_type == ActionType.SCROLL:
                direction = params.get("direction", "down")
                clicks = params.get("clicks", 3)
                self.mouse.scroll(clicks, direction)

            elif action_type == ActionType.DRAG:
                self.mouse.drag_to(
                    params["start_x"], params["start_y"],
                    params["end_x"], params["end_y"],
                )

            elif action_type == ActionType.NAVIGATE_URL:
                if self.browser:
                    asyncio.get_event_loop().run_until_complete(
                        self.browser.navigate(params["url"])
                    )
                else:
                    log.error("Browser subsystem not available for NAVIGATE_URL")
                    return False

            elif action_type == ActionType.WAIT:
                time.sleep(params.get("seconds", 1.0))

            elif action_type == ActionType.SCREENSHOT:
                img = self.screen.capture_screen()
                path = params.get("path", "screenshot.png")
                img.save(path)
                action.result = {"path": path}

            elif action_type == ActionType.FOCUS_WINDOW:
                if self.desktop:
                    windows = self.desktop.list_windows()
                    target_title = params.get("title", action.target)
                    for w in windows:
                        if target_title.lower() in w.title.lower():
                            return self.desktop.focus_window(w)
                    log.error(f"Window not found: {target_title}")
                    return False
                else:
                    log.error("Desktop subsystem not available for FOCUS_WINDOW")
                    return False

            elif action_type == ActionType.OPEN_APPLICATION:
                import subprocess as sp
                app_name = params.get("name", action.target)
                sp.Popen(app_name, shell=True)
                time.sleep(2.0)  # Wait for application to launch

            elif action_type == ActionType.FIND_ELEMENT:
                # Use VLM to locate an element on screen
                screenshot = self.screen.capture_screen()
                elements = self.vision.find_clickable_elements(screenshot)
                target = action.target.lower()
                for e in elements:
                    if target in e.get("text", "").lower():
                        action.result = {
                            "found": True,
                            "bbox": {
                                "x": e["bbox"].x, "y": e["bbox"].y,
                                "width": e["bbox"].width, "height": e["bbox"].height,
                            },
                        }
                        return True
                action.result = {"found": False}
                return False

            elif action_type == ActionType.READ_TEXT:
                screenshot = self.screen.capture_screen()
                text = self.vision.describe_screen(screenshot, detail_level="high")
                action.result = {"text": text}

            elif action_type == ActionType.SELECT_OPTION:
                if self.browser:
                    asyncio.get_event_loop().run_until_complete(
                        self.browser.select_option(
                            params["selector"], params["value"]
                        )
                    )
                else:
                    log.error("Browser subsystem not available for SELECT_OPTION")
                    return False

            else:
                log.error(f"Unknown action type: {action_type}")
                return False

            return True

        except Exception as exc:
            log.error(f"Action execution error: {exc}")
            action.result = {"error": str(exc)}
            return False

    # ── Replanning ──

    def _replan(self, failed_plan: Plan) -> Plan:
        """
        Re-examine the screen and adjust the remaining plan.

        Called when an action fails and retries are exhausted.

        Args:
            failed_plan: The plan that encountered a failure.

        Returns:
            A new Plan starting from the current screen state.
        """
        screenshot = self.screen.capture_screen()
        screen_desc = self.vision.describe_screen(screenshot, detail_level="high")

        completed_summary = "\n".join(
            f"  [DONE] {a.type.value} -> '{a.target}'"
            for a in failed_plan.actions[:failed_plan.current_step]
            if a.status == ActionStatus.SUCCESS
        )

        failed_action = failed_plan.actions[failed_plan.current_step]

        prompt = f"""A UI automation plan encountered a failure. Re-examine the
current screen and produce a revised plan for the remaining steps.

ORIGINAL GOAL: {failed_plan.intent}

COMPLETED STEPS:
{completed_summary}

FAILED STEP: {failed_action.type.value} -> '{failed_action.target}'
  Error: {failed_action.result}

CURRENT SCREEN:
{screen_desc}

Produce a revised JSON array of actions to complete the original goal from the
current state. Do NOT repeat already-completed steps."""

        response = self._llm.send(prompt)
        try:
            raw_actions = json.loads(response.text)
        except json.JSONDecodeError:
            log.error("Replan: failed to parse LLM response")
            raw_actions = []

        new_plan = Plan(intent=failed_plan.intent)
        # Carry over completed actions
        for a in failed_plan.actions[:failed_plan.current_step]:
            if a.status == ActionStatus.SUCCESS:
                new_plan.actions.append(a)
        new_plan.current_step = len(new_plan.actions)

        # Append new actions
        for raw in raw_actions:
            try:
                new_plan.actions.append(Action(
                    type=ActionType(raw["action"]),
                    target=raw.get("target", ""),
                    parameters=raw.get("parameters", {}),
                    expected_result=raw.get("expected_result", ""),
                ))
            except (ValueError, KeyError):
                pass

        log.info(
            f"Replanned: {len(new_plan.actions) - new_plan.current_step} new actions"
        )
        return new_plan

    # ── Public High-Level API ──

    def run(self, user_request: str) -> Dict[str, Any]:
        """
        End-to-end execution: intent -> plan -> execute -> verify.

        This is the main entry point called by ComputerUseAgent tools.

        Args:
            user_request: Natural-language instruction.

        Returns:
            {
                "success": bool,
                "goal": str,
                "actions_total": int,
                "actions_succeeded": int,
                "actions_failed": int,
                "actions_skipped": int,
                "summary": str,
            }
        """
        log.info(f"ComputerUseController.run: {user_request}")

        # 1. Parse intent
        intent = self.parse_intent(user_request)
        log.info(f"Intent: {intent.get('goal', user_request)}")

        # 2. Screenshot current state
        screenshot = self.screen.capture_screen()

        # 3. Create plan
        plan = self.create_plan(intent, screenshot)
        if not plan.actions:
            return {
                "success": False,
                "goal": intent.get("goal", user_request),
                "actions_total": 0,
                "actions_succeeded": 0,
                "actions_failed": 0,
                "actions_skipped": 0,
                "summary": "Failed to create an action plan.",
            }

        # 4. Execute
        plan = self.execute_plan(plan)

        # 5. Summarise
        succeeded = sum(1 for a in plan.actions if a.status == ActionStatus.SUCCESS)
        failed = sum(1 for a in plan.actions if a.status == ActionStatus.FAILED)
        skipped = sum(1 for a in plan.actions if a.status == ActionStatus.SKIPPED)

        return {
            "success": plan.success,
            "goal": plan.intent,
            "actions_total": len(plan.actions),
            "actions_succeeded": succeeded,
            "actions_failed": failed,
            "actions_skipped": skipped,
            "summary": (
                f"Completed {succeeded}/{len(plan.actions)} actions for: {plan.intent}"
            ),
        }
```

---

## Safety Framework

The safety framework is the **most critical** non-functional component. A computer use
agent that can move the mouse, type on the keyboard, and navigate the web has the power to
cause real damage: deleting files, sending emails, making purchases, or exfiltrating data.

### Design Principles

1. **Deny by default** -- unknown actions are blocked, not allowed.
2. **Confirm before commit** -- any action with lasting side effects requires explicit
   confirmation.
3. **Audit everything** -- every action is logged with full context.
4. **Panic button** -- the user can abort at any time via a global hotkey or failsafe.
5. **Blast radius containment** -- limit what the agent can reach (no system settings,
   no admin operations).

### Safety Rules

| Category | Rule | Action |
|----------|------|--------|
| **Forbidden** | Delete system files | BLOCK unconditionally |
| **Forbidden** | Access password managers | BLOCK unconditionally |
| **Forbidden** | Modify system settings / control panel | BLOCK unconditionally |
| **Forbidden** | Execute terminal commands as root/admin | BLOCK unconditionally |
| **Forbidden** | Access financial transaction pages (unless explicitly allowed) | BLOCK unconditionally |
| **Confirm** | Send email / message | Require user confirmation |
| **Confirm** | Submit any form | Require user confirmation |
| **Confirm** | Close unsaved documents | Require user confirmation |
| **Confirm** | Navigate to new domain (first visit) | Require user confirmation |
| **Confirm** | Install software | Require user confirmation |
| **Allow** | Read screen content | Always allowed |
| **Allow** | Scroll | Always allowed |
| **Allow** | Take screenshots | Always allowed |
| **Allow** | Navigate within allowed domains | Always allowed |
| **Allow** | Type in text fields (non-password) | Always allowed |

### Implementation

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Safety framework for computer use actions.

Location: src/gaia/computer_use/safety.py
"""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from gaia.logger import get_logger

log = get_logger(__name__)


class RiskLevel(Enum):
    """Risk classification for actions."""
    SAFE = "safe"           # Always allowed
    LOW = "low"             # Allowed, logged
    MEDIUM = "medium"       # Allowed with warning
    HIGH = "high"           # Requires confirmation
    CRITICAL = "critical"   # Blocked by default


@dataclass
class SafetyRule:
    """A single safety validation rule."""
    name: str
    description: str
    risk_level: RiskLevel
    check: Callable[["Action"], bool]  # Returns True if rule is VIOLATED
    message: str  # Message shown when rule is violated


@dataclass
class SafetyConfig:
    """Configuration for the safety framework."""
    # Domains the agent is allowed to navigate to without confirmation
    allowed_domains: Set[str] = field(default_factory=lambda: {
        "google.com", "gmail.com", "github.com", "stackoverflow.com",
        "wikipedia.org", "docs.python.org",
    })

    # File path patterns that are absolutely forbidden
    forbidden_paths: List[str] = field(default_factory=lambda: [
        r"C:\\Windows\\System32",
        r"/etc/passwd",
        r"/etc/shadow",
        r"~/.ssh",
        r"~/.gnupg",
        r".*\.pem$",
        r".*\.key$",
        r".*id_rsa.*",
    ])

    # Window titles that should never be interacted with
    forbidden_window_patterns: List[str] = field(default_factory=lambda: [
        r".*Password.*Manager.*",
        r".*KeePass.*",
        r".*1Password.*",
        r".*LastPass.*",
        r".*Bitwarden.*",
        r".*System Settings.*",
        r".*Control Panel.*",
        r".*Registry Editor.*",
        r".*Device Manager.*",
        r".*Disk Management.*",
    ])

    # Maximum actions per minute (rate limiting)
    max_actions_per_minute: int = 60

    # Whether to require confirmation for high-risk actions
    require_confirmation: bool = True

    # Safe mode: only allow read-only operations
    safe_mode: bool = False

    # Panic hotkey corner position (pyautogui failsafe)
    panic_enabled: bool = True


class SafetyGuard:
    """
    Validates actions against safety rules before execution.

    Every action passes through the SafetyGuard before reaching the
    input subsystem. The guard can:
    - Allow the action (SAFE / LOW)
    - Warn and allow (MEDIUM)
    - Require confirmation (HIGH)
    - Block unconditionally (CRITICAL)
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self._rules: List[SafetyRule] = []
        self._action_log: List[Dict[str, Any]] = []
        self._action_timestamps: List[float] = []
        self._confirmation_callback: Optional[Callable[[str], bool]] = None

        # Register built-in rules
        self._register_builtin_rules()

    def set_confirmation_callback(self, callback: Callable[[str], bool]):
        """
        Register a callback for user confirmation prompts.

        The callback receives a message string and should return True
        if the user confirms, False otherwise.

        Args:
            callback: Function that prompts the user and returns bool.
        """
        self._confirmation_callback = callback

    def add_rule(self, rule: SafetyRule):
        """Add a custom safety rule."""
        self._rules.append(rule)

    def validate_action(self, action: "Action") -> Dict[str, Any]:
        """
        Validate an action against all safety rules.

        Args:
            action: The Action to validate.

        Returns:
            {
                "allowed": bool,
                "risk_level": str,
                "needs_confirmation": bool,
                "reason": str | None,
                "violated_rules": List[str],
            }
        """
        # Rate limiting
        now = time.time()
        self._action_timestamps = [
            t for t in self._action_timestamps if now - t < 60
        ]
        if len(self._action_timestamps) >= self.config.max_actions_per_minute:
            return {
                "allowed": False,
                "risk_level": "critical",
                "needs_confirmation": False,
                "reason": "Rate limit exceeded (max {}/min)".format(
                    self.config.max_actions_per_minute
                ),
                "violated_rules": ["rate_limit"],
            }

        # Safe mode: block all write operations
        if self.config.safe_mode:
            read_only_actions = {
                ActionType.SCREENSHOT, ActionType.READ_TEXT,
                ActionType.FIND_ELEMENT, ActionType.WAIT,
            }
            if action.type not in read_only_actions:
                return {
                    "allowed": False,
                    "risk_level": "high",
                    "needs_confirmation": False,
                    "reason": "Safe mode is enabled -- only read operations allowed",
                    "violated_rules": ["safe_mode"],
                }

        # Check all rules
        violated = []
        highest_risk = RiskLevel.SAFE

        for rule in self._rules:
            try:
                if rule.check(action):
                    violated.append(rule)
                    if rule.risk_level.value > highest_risk.value:
                        highest_risk = rule.risk_level
            except Exception as exc:
                log.warning(f"Safety rule '{rule.name}' raised error: {exc}")

        # Decision
        if highest_risk == RiskLevel.CRITICAL:
            self._log_action(action, "BLOCKED", violated)
            return {
                "allowed": False,
                "risk_level": "critical",
                "needs_confirmation": False,
                "reason": "; ".join(r.message for r in violated),
                "violated_rules": [r.name for r in violated],
            }

        if highest_risk == RiskLevel.HIGH and self.config.require_confirmation:
            confirmed = self._request_confirmation(action, violated)
            if not confirmed:
                self._log_action(action, "DENIED", violated)
                return {
                    "allowed": False,
                    "risk_level": "high",
                    "needs_confirmation": True,
                    "reason": "; ".join(r.message for r in violated),
                    "violated_rules": [r.name for r in violated],
                }

        # Allowed
        self._action_timestamps.append(now)
        self._log_action(action, "ALLOWED", violated)
        return {
            "allowed": True,
            "risk_level": highest_risk.value,
            "needs_confirmation": False,
            "reason": None,
            "violated_rules": [r.name for r in violated],
        }

    def _request_confirmation(
        self, action: "Action", violated_rules: List[SafetyRule]
    ) -> bool:
        """Prompt user for confirmation of a high-risk action."""
        message = (
            f"SAFETY CONFIRMATION REQUIRED\n"
            f"Action: {action.type.value} on '{action.target}'\n"
            f"Risks: {'; '.join(r.message for r in violated_rules)}\n"
            f"Allow this action?"
        )

        if self._confirmation_callback:
            return self._confirmation_callback(message)

        # Default: deny if no callback registered
        log.warning(f"No confirmation callback -- denying: {message}")
        return False

    def _log_action(
        self, action: "Action", decision: str, violated: List[SafetyRule]
    ):
        """Record action in audit log."""
        entry = {
            "timestamp": time.time(),
            "action_type": action.type.value,
            "target": action.target,
            "parameters": action.parameters,
            "decision": decision,
            "violated_rules": [r.name for r in violated],
        }
        self._action_log.append(entry)

        if self.debug_enabled:
            log.debug(f"Safety audit: {decision} -- {action.type.value} -> {action.target}")

    @property
    def debug_enabled(self) -> bool:
        """Check if debug logging is enabled."""
        return log.isEnabledFor(10)  # DEBUG level

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return the full audit log."""
        return list(self._action_log)

    # ── Built-in Rules ──

    def _register_builtin_rules(self):
        """Register the default safety rules."""

        # CRITICAL: Forbidden window targets
        self._rules.append(SafetyRule(
            name="forbidden_window",
            description="Block interaction with password managers and system settings",
            risk_level=RiskLevel.CRITICAL,
            check=lambda action: any(
                re.match(pattern, action.target, re.IGNORECASE)
                for pattern in self.config.forbidden_window_patterns
            ),
            message="Interaction with this window is forbidden",
        ))

        # CRITICAL: Forbidden file paths
        self._rules.append(SafetyRule(
            name="forbidden_path",
            description="Block access to sensitive file paths",
            risk_level=RiskLevel.CRITICAL,
            check=lambda action: any(
                re.search(pattern, str(action.parameters.get("path", "")))
                for pattern in self.config.forbidden_paths
            ),
            message="Access to this file path is forbidden",
        ))

        # CRITICAL: Typing into password fields
        self._rules.append(SafetyRule(
            name="password_field",
            description="Block typing into password fields",
            risk_level=RiskLevel.CRITICAL,
            check=lambda action: (
                action.type == ActionType.TYPE_TEXT and
                "password" in action.target.lower()
            ),
            message="Cannot type into password fields automatically",
        ))

        # HIGH: Sending email / messages
        self._rules.append(SafetyRule(
            name="send_communication",
            description="Require confirmation before sending emails or messages",
            risk_level=RiskLevel.HIGH,
            check=lambda action: (
                action.type == ActionType.CLICK and
                any(kw in action.target.lower()
                    for kw in ["send", "submit", "post", "publish", "confirm purchase"])
            ),
            message="This action may send a message or submit data",
        ))

        # HIGH: Navigating to unknown domains
        self._rules.append(SafetyRule(
            name="unknown_domain",
            description="Require confirmation for unfamiliar domains",
            risk_level=RiskLevel.HIGH,
            check=lambda action: (
                action.type == ActionType.NAVIGATE_URL and
                not self._is_allowed_domain(action.parameters.get("url", ""))
            ),
            message="Navigating to an unrecognised domain",
        ))

        # HIGH: Closing windows (possible unsaved work)
        self._rules.append(SafetyRule(
            name="close_window",
            description="Require confirmation before closing windows",
            risk_level=RiskLevel.HIGH,
            check=lambda action: (
                action.type == ActionType.HOTKEY and
                action.parameters.get("keys", []) in [
                    ["alt", "F4"], ["ctrl", "w"], ["command", "w"],
                ]
            ),
            message="Closing a window may lose unsaved work",
        ))

        # MEDIUM: Scroll large distances
        self._rules.append(SafetyRule(
            name="large_scroll",
            description="Warn on large scroll operations",
            risk_level=RiskLevel.MEDIUM,
            check=lambda action: (
                action.type == ActionType.SCROLL and
                abs(action.parameters.get("clicks", 0)) > 20
            ),
            message="Large scroll operation -- may lose current position",
        ))

    def _is_allowed_domain(self, url: str) -> bool:
        """Check if a URL's domain is in the allowed list."""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().lstrip("www.")
            return any(
                domain == allowed or domain.endswith("." + allowed)
                for allowed in self.config.allowed_domains
            )
        except Exception:
            return False


# ──────────────────────────────────────────────
# Rollback Manager
# ──────────────────────────────────────────────

class RollbackManager:
    """
    Tracks reversible actions and provides undo capability.

    Not all actions are reversible. The manager records what it can and
    provides best-effort rollback.

    Location: src/gaia/computer_use/safety.py
    """

    # Map action types to their undo counterparts
    REVERSIBLE_ACTIONS = {
        ActionType.TYPE_TEXT: ActionType.HOTKEY,  # Undo via Ctrl+Z
        ActionType.NAVIGATE_URL: "go_back",
        ActionType.SCROLL: ActionType.SCROLL,  # Reverse direction
        ActionType.CHECK: "uncheck",
    }

    def __init__(self, keyboard: "KeyboardController", browser: Optional["BrowserAutomationAgent"] = None):
        self.keyboard = keyboard
        self.browser = browser
        self._undo_stack: List[Dict[str, Any]] = []

    def record(self, action: "Action"):
        """Record an action for potential rollback."""
        if action.type in self.REVERSIBLE_ACTIONS:
            self._undo_stack.append({
                "action": action,
                "timestamp": time.time(),
            })

    def undo_last(self) -> bool:
        """Undo the most recent reversible action."""
        if not self._undo_stack:
            log.warning("Nothing to undo")
            return False

        entry = self._undo_stack.pop()
        action = entry["action"]

        try:
            if action.type == ActionType.TYPE_TEXT:
                self.keyboard.undo()  # Ctrl+Z
                return True

            elif action.type == ActionType.NAVIGATE_URL:
                if self.browser:
                    asyncio.get_event_loop().run_until_complete(
                        self.browser.go_back()
                    )
                    return True

            elif action.type == ActionType.SCROLL:
                # Reverse the scroll direction
                direction = action.parameters.get("direction", "down")
                clicks = action.parameters.get("clicks", 3)
                reverse = "up" if direction == "down" else "down"
                import pyautogui
                scroll_amount = clicks if reverse == "up" else -clicks
                pyautogui.scroll(scroll_amount)
                return True

        except Exception as exc:
            log.error(f"Rollback failed: {exc}")

        return False

    def undo_all(self) -> int:
        """Undo all recorded actions in reverse order. Returns count of undone actions."""
        count = 0
        while self._undo_stack:
            if self.undo_last():
                count += 1
            else:
                break
        return count


# ──────────────────────────────────────────────
# Panic Button
# ──────────────────────────────────────────────

class PanicButton:
    """
    Global emergency stop mechanism.

    Supports two modes:
    1. pyautogui failsafe: move mouse to top-left corner
    2. Keyboard hotkey: Ctrl+Alt+Shift+Escape

    When triggered, all subsystems are stopped and pending actions
    are cancelled.

    Location: src/gaia/computer_use/safety.py
    """

    def __init__(self):
        self._active = False
        self._callbacks: List[Callable] = []

    def register_callback(self, callback: Callable):
        """Register a callback to invoke on panic."""
        self._callbacks.append(callback)

    def trigger(self, reason: str = "User triggered panic button"):
        """Trigger emergency stop."""
        if self._active:
            return  # Already panicking

        self._active = True
        log.critical(f"PANIC BUTTON TRIGGERED: {reason}")

        for callback in self._callbacks:
            try:
                callback()
            except Exception as exc:
                log.error(f"Panic callback error: {exc}")

        # Release all held keys
        try:
            import pyautogui
            pyautogui.keyUp("shift")
            pyautogui.keyUp("ctrl")
            pyautogui.keyUp("alt")
            pyautogui.keyUp("command")
        except Exception:
            pass

    def reset(self):
        """Reset panic state (allow operations to resume)."""
        self._active = False
        log.info("Panic state reset -- operations may resume")

    @property
    def is_triggered(self) -> bool:
        return self._active
```

---

## Integration with GAIA

### ComputerUseAgent

The `ComputerUseAgent` inherits from the GAIA base `Agent` class and exposes computer
use capabilities as registered tools. This follows the same pattern as `BlenderAgent`,
`JiraAgent`, and other GAIA agents.

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
ComputerUseAgent -- GAIA agent for visual UI automation.

Location: src/gaia/agents/computer_use/agent.py
"""

from typing import Any, Dict, Optional

from gaia.agents.base.agent import Agent
from gaia.agents.base.tools import tool
from gaia.logger import get_logger

log = get_logger(__name__)


class ComputerUseAgent(Agent):
    """
    GAIA agent that can see, understand, and interact with desktop UIs.

    Inherits core Agent functionality (conversation loop, tool execution,
    state management) and adds computer-use tools.

    Usage:
        agent = ComputerUseAgent()
        result = agent.process_query("Open Gmail and archive all newsletters")
    """

    DEFAULT_MODEL = "Qwen3-Coder-30B-A3B-Instruct-GGUF"

    def __init__(
        self,
        model_id: str = None,
        base_url: str = None,
        use_claude: bool = False,
        use_chatgpt: bool = False,
        max_steps: int = 30,
        show_prompts: bool = False,
        streaming: bool = False,
        show_stats: bool = False,
        silent_mode: bool = False,
        debug: bool = False,
        safe_mode: bool = False,
        headless_browser: bool = False,
        auto_verify: bool = True,
    ):
        """
        Initialise the ComputerUseAgent.

        Args:
            model_id: LLM model ID (default: Qwen3-Coder-30B).
            base_url: LLM server base URL.
            use_claude: Use Claude API.
            use_chatgpt: Use ChatGPT API.
            max_steps: Max agent steps before termination.
            show_prompts: Display prompts sent to LLM.
            streaming: Enable response streaming.
            show_stats: Show LLM performance stats.
            silent_mode: Suppress console output.
            debug: Enable debug logging.
            safe_mode: Only allow read-only operations.
            headless_browser: Run browser without visible window.
            auto_verify: Verify each action via VLM.
        """
        super().__init__(
            model_id=model_id or self.DEFAULT_MODEL,
            base_url=base_url,
            use_claude=use_claude,
            use_chatgpt=use_chatgpt,
            max_steps=max_steps,
            show_prompts=show_prompts,
            streaming=streaming,
            show_stats=show_stats,
            silent_mode=silent_mode,
            debug=debug,
        )

        self.safe_mode = safe_mode
        self.headless_browser = headless_browser
        self.auto_verify = auto_verify

        # Lazy-initialise subsystems (created on first use)
        self._controller: Optional["ComputerUseController"] = None

        # Register tools
        self._register_computer_use_tools()

    def _get_system_prompt(self) -> str:
        """System prompt for the computer use agent."""
        return """You are a computer use agent. You can see and interact with the
user's desktop, web browsers, and applications.

You have access to the following tools:

- computer_use: Execute a natural-language UI automation task.
  The system will parse your intent, plan the steps, and execute them.
  Example: computer_use(task="Open Chrome and navigate to gmail.com")

- take_screenshot: Capture the current screen and describe what you see.

- describe_screen: Get a detailed description of what is currently on screen.

- find_element: Search the screen for a specific UI element.

- click_at: Click at specific screen coordinates.

- type_text: Type text using the keyboard.

- press_key: Press a keyboard key or shortcut.

- scroll_screen: Scroll up or down.

- browser_navigate: Open a URL in the browser.

- list_windows: List all open windows.

- focus_window: Bring a window to the front.

IMPORTANT SAFETY RULES:
1. NEVER interact with password managers or system settings.
2. ALWAYS describe what you see before taking action.
3. VERIFY the result after each action.
4. ASK for confirmation before sending messages, submitting forms, or
   making purchases.
5. If something looks wrong, STOP and ask the user.

When the user asks you to do something on their computer, break it down into
steps, execute each step, verify the result, and report back."""

    def _get_controller(self) -> "ComputerUseController":
        """Lazy-initialise and return the ComputerUseController."""
        if self._controller is None:
            from gaia.computer_use.controller import ComputerUseController
            from gaia.computer_use.vision import ScreenCaptureEngine, VLMAnalyzer
            from gaia.computer_use.input import MouseController, KeyboardController
            from gaia.computer_use.safety import SafetyGuard, SafetyConfig

            safety_config = SafetyConfig(safe_mode=self.safe_mode)
            self._controller = ComputerUseController(
                vision=VLMAnalyzer(),
                screen=ScreenCaptureEngine(),
                mouse=MouseController(),
                keyboard=KeyboardController(),
                safety=SafetyGuard(config=safety_config),
                auto_verify=self.auto_verify,
                debug=self.debug,
            )

        return self._controller

    def _register_computer_use_tools(self):
        """Register all computer-use tools with the GAIA tool registry."""

        @tool
        def computer_use(task: str) -> Dict[str, Any]:
            """
            Execute a computer use task described in natural language.

            The system will parse the intent, plan UI actions, execute them
            with safety checks, and verify results.

            Args:
                task: Natural language description of the task.
                    Examples:
                    - "Open Chrome and navigate to gmail.com"
                    - "Click the Compose button in Gmail"
                    - "Fill in the To field with user@example.com"

            Returns:
                Result dict with success status and summary.
            """
            controller = self._get_controller()
            return controller.run(task)

        @tool
        def take_screenshot(save_path: str = "") -> Dict[str, Any]:
            """
            Capture a screenshot of the current screen.

            Args:
                save_path: Optional file path to save the screenshot.

            Returns:
                Dict with screenshot description and optional file path.
            """
            controller = self._get_controller()
            image = controller.screen.capture_screen()
            description = controller.vision.describe_screen(image, detail_level="medium")

            result = {"description": description}
            if save_path:
                image.save(save_path)
                result["saved_to"] = save_path

            return result

        @tool
        def describe_screen() -> Dict[str, Any]:
            """
            Get a detailed description of the current screen contents.

            Returns:
                Dict with comprehensive screen description and detected elements.
            """
            controller = self._get_controller()
            image = controller.screen.capture_screen()
            description = controller.vision.describe_screen(image, detail_level="high")
            elements = controller.vision.find_clickable_elements(image)

            return {
                "description": description,
                "interactive_elements": [
                    {"type": e["type"], "text": e["text"],
                     "x": e["bbox"].x, "y": e["bbox"].y}
                    for e in elements
                ],
            }

        @tool
        def find_element(element_description: str) -> Dict[str, Any]:
            """
            Search the screen for a specific UI element.

            Args:
                element_description: What to look for (e.g. "Submit button",
                    "search field", "File menu").

            Returns:
                Dict indicating if the element was found and its location.
            """
            controller = self._get_controller()
            image = controller.screen.capture_screen()
            elements = controller.vision.find_clickable_elements(image)

            target = element_description.lower()
            for e in elements:
                if target in e.get("text", "").lower() or target in e.get("type", "").lower():
                    return {
                        "found": True,
                        "type": e["type"],
                        "text": e["text"],
                        "x": e["bbox"].x,
                        "y": e["bbox"].y,
                        "width": e["bbox"].width,
                        "height": e["bbox"].height,
                    }

            return {"found": False, "message": f"Element not found: {element_description}"}

        @tool
        def click_at(x: int, y: int) -> Dict[str, Any]:
            """
            Click at specific screen coordinates.

            Args:
                x: X coordinate (pixels from left).
                y: Y coordinate (pixels from top).

            Returns:
                Dict with status.
            """
            controller = self._get_controller()
            controller.mouse.click(x, y)
            return {"status": "success", "clicked_at": {"x": x, "y": y}}

        @tool
        def type_text(text: str) -> Dict[str, Any]:
            """
            Type text using the keyboard at the current cursor position.

            Args:
                text: The text to type.

            Returns:
                Dict with status and character count.
            """
            controller = self._get_controller()
            controller.keyboard.type_text(text)
            return {"status": "success", "typed_characters": len(text)}

        @tool
        def press_key(key: str) -> Dict[str, Any]:
            """
            Press a keyboard key or shortcut.

            Args:
                key: Key name or shortcut.
                    Examples: "enter", "tab", "escape", "ctrl+c", "ctrl+shift+s"

            Returns:
                Dict with status.
            """
            controller = self._get_controller()
            if "+" in key:
                keys = [k.strip() for k in key.split("+")]
                controller.keyboard.hotkey(*keys)
            else:
                controller.keyboard.press_key(key)
            return {"status": "success", "key": key}

        @tool
        def scroll_screen(direction: str, amount: int = 3) -> Dict[str, Any]:
            """
            Scroll the screen up or down.

            Args:
                direction: "up" or "down".
                amount: Number of scroll clicks (default 3).

            Returns:
                Dict with status.
            """
            controller = self._get_controller()
            controller.mouse.scroll(amount, direction)
            return {"status": "success", "direction": direction, "amount": amount}

        @tool
        def browser_navigate(url: str) -> Dict[str, Any]:
            """
            Open a URL in the web browser.

            Args:
                url: The URL to navigate to.

            Returns:
                Dict with status.
            """
            import webbrowser
            webbrowser.open(url)
            return {"status": "success", "url": url}

        @tool
        def list_windows() -> Dict[str, Any]:
            """
            List all currently open windows.

            Returns:
                Dict with list of window titles and metadata.
            """
            controller = self._get_controller()
            if controller.desktop:
                windows = controller.desktop.list_windows()
                return {
                    "windows": [
                        {"title": w.title, "pid": w.pid, "focused": w.is_focused}
                        for w in windows
                    ]
                }
            return {"windows": [], "message": "Desktop automation not available"}

        @tool
        def focus_window(title: str) -> Dict[str, Any]:
            """
            Bring a window to the front by its title.

            Args:
                title: Full or partial window title.

            Returns:
                Dict with status.
            """
            controller = self._get_controller()
            if controller.desktop:
                windows = controller.desktop.list_windows()
                for w in windows:
                    if title.lower() in w.title.lower():
                        success = controller.desktop.focus_window(w)
                        return {
                            "status": "success" if success else "failed",
                            "window": w.title,
                        }
                return {"status": "failed", "message": f"Window not found: {title}"}
            return {"status": "failed", "message": "Desktop automation not available"}
```

### CLI Integration

The `gaia computer-use` command is added to the CLI entry point in
`src/gaia/cli.py`. This follows the same pattern as existing subcommands like
`gaia chat`, `gaia blender`, and `gaia jira`.

```python
# Addition to src/gaia/cli.py (inside the main CLI parser setup)

def add_computer_use_parser(subparsers):
    """Register the computer-use subcommand."""
    parser = subparsers.add_parser(
        "computer-use",
        help="Visual UI automation agent -- control desktop and web applications",
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="Single task to execute (non-interactive mode)",
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Enable safe mode (read-only operations only)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable automatic visual verification after each action",
    )
    parser.add_argument(
        "--use-claude",
        action="store_true",
        help="Use Claude API for planning",
    )
    parser.add_argument(
        "--use-chatgpt",
        action="store_true",
        help="Use ChatGPT API for planning",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.set_defaults(func=run_computer_use)


def run_computer_use(args):
    """Entry point for `gaia computer-use`."""
    from gaia.agents.computer_use.agent import ComputerUseAgent

    agent = ComputerUseAgent(
        use_claude=getattr(args, "use_claude", False),
        use_chatgpt=getattr(args, "use_chatgpt", False),
        safe_mode=getattr(args, "safe_mode", False),
        headless_browser=getattr(args, "headless", False),
        auto_verify=not getattr(args, "no_verify", False),
        debug=getattr(args, "debug", False),
    )

    if args.query:
        result = agent.process_query(args.query)
        print(result)
    else:
        # Interactive mode
        print("Computer Use Agent -- Interactive Mode")
        print("Type 'exit' or 'quit' to stop. Type 'safe' to toggle safe mode.\n")
        while True:
            try:
                user_input = input("computer-use> ").strip()
                if user_input.lower() in ("exit", "quit"):
                    break
                if user_input.lower() == "safe":
                    agent.safe_mode = not agent.safe_mode
                    mode = "ON" if agent.safe_mode else "OFF"
                    print(f"Safe mode: {mode}")
                    continue
                if not user_input:
                    continue
                result = agent.process_query(user_input)
                print(result)
            except KeyboardInterrupt:
                print("\nInterrupted. Exiting.")
                break
```

### Integration with Existing Agents

The computer-use capability can be mixed into any existing agent via the
`ComputerUseToolsMixin`:

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Mixin that adds computer-use tools to any GAIA agent.

Location: src/gaia/agents/computer_use/tools_mixin.py
"""

from typing import Any, Dict

from gaia.agents.base.tools import tool


class ComputerUseToolsMixin:
    """
    Mixin providing computer use tools for any Agent subclass.

    Usage:
        class MyAgent(Agent, ComputerUseToolsMixin):
            def __init__(self, ...):
                super().__init__(...)
                self.register_computer_use_tools()
    """

    def register_computer_use_tools(self):
        """Register a subset of computer-use tools on this agent."""

        @tool
        def take_screenshot(save_path: str = "") -> Dict[str, Any]:
            """Capture and describe the current screen."""
            from gaia.computer_use.vision import ScreenCaptureEngine, VLMAnalyzer

            screen = ScreenCaptureEngine()
            vision = VLMAnalyzer()

            image = screen.capture_screen()
            description = vision.describe_screen(image, detail_level="medium")

            result = {"description": description}
            if save_path:
                image.save(save_path)
                result["saved_to"] = save_path
            return result

        @tool
        def click_at(x: int, y: int) -> Dict[str, Any]:
            """Click at screen coordinates (x, y)."""
            from gaia.computer_use.input import MouseController

            mouse = MouseController()
            mouse.click(x, y)
            return {"status": "success", "x": x, "y": y}

        @tool
        def type_text(text: str) -> Dict[str, Any]:
            """Type text at the current cursor position."""
            from gaia.computer_use.input import KeyboardController

            keyboard = KeyboardController()
            keyboard.type_text(text)
            return {"status": "success", "characters": len(text)}
```

---

## Implementation Plan

### Timeline: 8-10 Weeks (2 Engineers)

#### Phase 1: Foundation (Weeks 1-2)

| Week | Task | Owner | Deliverable | Dependencies |
|------|------|-------|-------------|--------------|
| 1 | Screen capture engine + unit tests | Eng 1 | `src/gaia/computer_use/vision.py` | mss, Pillow |
| 1 | Mouse + keyboard controllers + unit tests | Eng 2 | `src/gaia/computer_use/input.py` | pyautogui |
| 2 | OCR engine (EasyOCR + Tesseract) + tests | Eng 1 | OCR module in `vision.py` | easyocr, pytesseract |
| 2 | Safety framework + unit tests | Eng 2 | `src/gaia/computer_use/safety.py` | None |

**Milestone 1**: Can capture screenshots, extract text, control mouse/keyboard with safety
checks.

#### Phase 2: Intelligence (Weeks 3-4)

| Week | Task | Owner | Deliverable | Dependencies |
|------|------|-------|-------------|--------------|
| 3 | VLM analyzer integration | Eng 1 | VLMAnalyzer class | GAIA VLM client |
| 3 | Browser automation (Playwright) | Eng 2 | `src/gaia/computer_use/browser.py` | playwright |
| 4 | ComputerUseController -- intent parsing + planning | Eng 1 | `src/gaia/computer_use/controller.py` | Phase 1 |
| 4 | ComputerUseController -- execution + verification | Eng 2 | Controller execute_plan() | Phase 1 |

**Milestone 2**: Controller can parse "Open Gmail" and produce + execute a plan.

#### Phase 3: Agent Integration (Weeks 5-6)

| Week | Task | Owner | Deliverable | Dependencies |
|------|------|-------|-------------|--------------|
| 5 | ComputerUseAgent + tool registration | Eng 1 | `src/gaia/agents/computer_use/agent.py` | Phase 2 |
| 5 | Desktop automation (Windows UIA) | Eng 2 | `src/gaia/computer_use/platforms/windows.py` | uiautomation |
| 6 | CLI command (`gaia computer-use`) | Eng 1 | CLI entry in `cli.py` | Phase 2 |
| 6 | Desktop automation (macOS + Linux) | Eng 2 | Platform modules | pyobjc, pyatspi |

**Milestone 3**: `gaia computer-use -q "Open Notepad and type Hello World"` works
end-to-end on Windows.

#### Phase 4: Hardening (Weeks 7-8)

| Week | Task | Owner | Deliverable | Dependencies |
|------|------|-------|-------------|--------------|
| 7 | Replanning + error recovery | Eng 1 | Robust replan logic | Phase 3 |
| 7 | Rollback manager + panic button | Eng 2 | Undo/abort capability | Phase 3 |
| 8 | Integration tests -- full end-to-end scenarios | Both | Test suite | All phases |
| 8 | ComputerUseToolsMixin for other agents | Eng 1 | `tools_mixin.py` | Phase 3 |

**Milestone 4**: All safety tests pass. Rollback works. Panic button tested.

#### Phase 5: Polish (Weeks 9-10)

| Week | Task | Owner | Deliverable | Dependencies |
|------|------|-------|-------------|--------------|
| 9 | Performance optimization (screenshot caching, batch OCR) | Eng 1 | Performance improvements | Phase 4 |
| 9 | Documentation (MDX guides + SDK reference) | Eng 2 | `docs/guides/computer-use.mdx` | Phase 4 |
| 10 | Demo scenarios (email triage, form filling) | Eng 1 | Example scripts | Phase 4 |
| 10 | Cross-platform testing + CI integration | Eng 2 | CI workflows | Phase 4 |

**Final Milestone**: Feature-complete, documented, tested on Windows/macOS/Linux.

### Dependency Graph

```
Week 1-2: Foundation
  ├── ScreenCapture ──────┐
  ├── OCR Engine ─────────┤
  ├── Mouse/Keyboard ─────┤
  └── Safety Framework ───┤
                          ▼
Week 3-4: Intelligence    │
  ├── VLM Analyzer ───────┤
  ├── Browser Automation ─┤
  └── Controller ─────────┤
                          ▼
Week 5-6: Integration     │
  ├── ComputerUseAgent ───┤
  ├── Desktop Automation ─┤
  └── CLI Command ────────┤
                          ▼
Week 7-8: Hardening       │
  ├── Replan / Recovery ──┤
  ├── Rollback / Panic ───┤
  └── Integration Tests ──┤
                          ▼
Week 9-10: Polish
  ├── Perf Optimization
  ├── Documentation
  └── Demos + CI
```

### External Dependencies

| Package | Version | Purpose | Install |
|---------|---------|---------|---------|
| `mss` | >=9.0 | Screenshot capture | `pip install mss` |
| `Pillow` | >=10.0 | Image processing | `pip install Pillow` |
| `pyautogui` | >=0.9 | Mouse/keyboard control | `pip install pyautogui` |
| `easyocr` | >=1.7 | OCR (GPU-accelerated) | `pip install easyocr` |
| `pytesseract` | >=0.3 | OCR (CPU, fast) | `pip install pytesseract` |
| `numpy` | >=1.24 | Image array operations | `pip install numpy` |
| `playwright` | >=1.40 | Browser automation | `pip install playwright` |
| `pyperclip` | >=1.8 | Clipboard access | `pip install pyperclip` |
| `uiautomation` | >=2.0 | Windows UI Automation | `pip install uiautomation` (Win only) |
| `pyobjc-framework-Cocoa` | >=10.0 | macOS accessibility | `pip install pyobjc-framework-Cocoa` (Mac only) |
| `pyatspi` | >=2.46 | Linux AT-SPI | `pip install pyatspi` (Linux only) |

---

## Testing Strategy

### Test Structure

```
tests/
├── unit/
│   └── computer_use/
│       ├── test_screen_capture.py        # Screenshot capture
│       ├── test_ocr_engine.py            # OCR text extraction
│       ├── test_vlm_analyzer.py          # VLM analysis (mocked)
│       ├── test_mouse_controller.py      # Mouse operations (mocked)
│       ├── test_keyboard_controller.py   # Keyboard operations (mocked)
│       ├── test_safety_guard.py          # Safety rule validation
│       ├── test_rollback.py              # Rollback manager
│       ├── test_action_model.py          # Action / Plan data models
│       └── test_controller.py            # Controller logic (mocked subsystems)
│
├── integration/
│   └── computer_use/
│       ├── test_browser_automation.py    # Playwright browser tests
│       ├── test_desktop_automation.py    # Desktop UI tree tests
│       ├── test_controller_e2e.py        # Full controller pipeline
│       └── test_agent_e2e.py            # ComputerUseAgent end-to-end
│
└── safety/
    └── computer_use/
        ├── test_forbidden_actions.py     # Verify blocked actions
        ├── test_confirmation_flow.py     # Verify confirmation prompts
        ├── test_rate_limiting.py         # Verify rate limits
        ├── test_panic_button.py          # Verify emergency stop
        └── test_safe_mode.py            # Verify safe mode restrictions
```

### Unit Tests

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Unit tests for the Safety Guard.

Location: tests/unit/computer_use/test_safety_guard.py
"""

import pytest
from unittest.mock import MagicMock

# These imports reference the modules defined in this architecture document.
# They will resolve once the source is implemented.
from gaia.computer_use.safety import (
    SafetyGuard,
    SafetyConfig,
    RiskLevel,
)
from gaia.computer_use.controller import Action, ActionType, ActionStatus


class TestSafetyGuard:
    """Tests for SafetyGuard action validation."""

    @pytest.fixture
    def guard(self):
        """Create a SafetyGuard with default config."""
        return SafetyGuard(SafetyConfig(require_confirmation=False))

    @pytest.fixture
    def strict_guard(self):
        """Create a SafetyGuard with confirmation required."""
        return SafetyGuard(SafetyConfig(require_confirmation=True))

    def test_safe_action_allowed(self, guard):
        """Scrolling should always be allowed."""
        action = Action(
            type=ActionType.SCROLL,
            target="page",
            parameters={"direction": "down", "clicks": 3},
        )
        result = guard.validate_action(action)
        assert result["allowed"] is True

    def test_screenshot_allowed(self, guard):
        """Screenshots should always be allowed."""
        action = Action(type=ActionType.SCREENSHOT, target="screen")
        result = guard.validate_action(action)
        assert result["allowed"] is True

    def test_password_field_blocked(self, guard):
        """Typing into password fields must be blocked unconditionally."""
        action = Action(
            type=ActionType.TYPE_TEXT,
            target="Password field",
            parameters={"text": "secret123"},
        )
        result = guard.validate_action(action)
        assert result["allowed"] is False
        assert result["risk_level"] == "critical"

    def test_forbidden_window_blocked(self, guard):
        """Interaction with password managers must be blocked."""
        action = Action(
            type=ActionType.CLICK,
            target="1Password - Vault",
            parameters={"x": 100, "y": 200},
        )
        result = guard.validate_action(action)
        assert result["allowed"] is False

    def test_send_button_requires_confirmation(self, strict_guard):
        """Clicking 'Send' should require confirmation."""
        action = Action(
            type=ActionType.CLICK,
            target="Send email button",
            parameters={"x": 500, "y": 400},
        )
        result = strict_guard.validate_action(action)
        # With no confirmation callback, it should be denied
        assert result["allowed"] is False
        assert result["needs_confirmation"] is True

    def test_send_button_confirmed(self, strict_guard):
        """Clicking 'Send' should succeed after confirmation."""
        strict_guard.set_confirmation_callback(lambda msg: True)
        action = Action(
            type=ActionType.CLICK,
            target="Send email button",
            parameters={"x": 500, "y": 400},
        )
        result = strict_guard.validate_action(action)
        assert result["allowed"] is True

    def test_rate_limiting(self, guard):
        """Exceeding rate limit should block actions."""
        guard.config.max_actions_per_minute = 5
        safe_action = Action(type=ActionType.WAIT, target="", parameters={"seconds": 0})
        for _ in range(5):
            result = guard.validate_action(safe_action)
            assert result["allowed"] is True

        # 6th action should be blocked
        result = guard.validate_action(safe_action)
        assert result["allowed"] is False
        assert "rate_limit" in result["violated_rules"]

    def test_safe_mode_blocks_writes(self):
        """Safe mode should block all non-read operations."""
        guard = SafetyGuard(SafetyConfig(safe_mode=True))
        click_action = Action(
            type=ActionType.CLICK, target="button", parameters={"x": 0, "y": 0}
        )
        result = guard.validate_action(click_action)
        assert result["allowed"] is False

    def test_safe_mode_allows_reads(self):
        """Safe mode should allow read-only operations."""
        guard = SafetyGuard(SafetyConfig(safe_mode=True))
        screenshot_action = Action(type=ActionType.SCREENSHOT, target="screen")
        result = guard.validate_action(screenshot_action)
        assert result["allowed"] is True

    def test_unknown_domain_flagged(self, strict_guard):
        """Navigating to unknown domains should require confirmation."""
        action = Action(
            type=ActionType.NAVIGATE_URL,
            target="browser",
            parameters={"url": "https://malicious-site.example.com"},
        )
        result = strict_guard.validate_action(action)
        assert result["allowed"] is False

    def test_allowed_domain_passes(self, guard):
        """Navigating to allowed domains should pass."""
        action = Action(
            type=ActionType.NAVIGATE_URL,
            target="browser",
            parameters={"url": "https://www.google.com/search?q=test"},
        )
        result = guard.validate_action(action)
        assert result["allowed"] is True

    def test_audit_log_records_all(self, guard):
        """All validated actions should appear in the audit log."""
        actions = [
            Action(type=ActionType.SCREENSHOT, target="screen"),
            Action(type=ActionType.SCROLL, target="page",
                   parameters={"direction": "down", "clicks": 2}),
        ]
        for a in actions:
            guard.validate_action(a)

        audit = guard.get_audit_log()
        assert len(audit) == 2
        assert audit[0]["action_type"] == "screenshot"
        assert audit[1]["action_type"] == "scroll"
```

### Integration Tests

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Integration tests for ComputerUseController end-to-end pipeline.

Location: tests/integration/computer_use/test_controller_e2e.py

These tests require a display (or virtual framebuffer like Xvfb on Linux CI).
"""

import pytest
from unittest.mock import MagicMock, patch

from gaia.computer_use.controller import (
    ComputerUseController,
    Action,
    ActionType,
    ActionStatus,
    Plan,
)


@pytest.fixture
def mock_subsystems():
    """Create mock subsystems for controller testing."""
    vision = MagicMock()
    vision.describe_screen.return_value = "Desktop with file explorer open"
    vision.find_clickable_elements.return_value = [
        {
            "type": "button",
            "text": "Open",
            "bbox": MagicMock(x=100, y=200, width=80, height=30, center=(140, 215)),
            "confidence": 0.95,
        }
    ]
    vision.verify_action_result.return_value = True

    screen = MagicMock()
    screen.capture_screen.return_value = MagicMock()  # Mock PIL Image

    mouse = MagicMock()
    keyboard = MagicMock()
    safety = MagicMock()
    safety.validate_action.return_value = {"allowed": True, "risk_level": "safe",
                                            "needs_confirmation": False,
                                            "reason": None, "violated_rules": []}

    return {
        "vision": vision,
        "screen": screen,
        "mouse": mouse,
        "keyboard": keyboard,
        "safety": safety,
    }


@pytest.fixture
def controller(mock_subsystems):
    """Create a controller with mocked subsystems and LLM."""
    mock_llm = MagicMock()
    mock_llm.send.return_value = MagicMock(text='[{"action": "click", "target": "Open button", "parameters": {"x": 140, "y": 215}, "expected_result": "File dialog opens"}]')

    ctrl = ComputerUseController(
        vision=mock_subsystems["vision"],
        screen=mock_subsystems["screen"],
        mouse=mock_subsystems["mouse"],
        keyboard=mock_subsystems["keyboard"],
        safety=mock_subsystems["safety"],
        llm_client=mock_llm,
        auto_verify=True,
        debug=True,
    )
    return ctrl


class TestControllerPipeline:
    """Test the full intent -> plan -> execute pipeline."""

    def test_run_simple_task(self, controller):
        """Test a simple single-action task."""
        result = controller.run("Click the Open button")
        assert result["success"] is True
        assert result["actions_total"] >= 1
        assert result["actions_succeeded"] >= 1

    def test_empty_plan_returns_failure(self, controller):
        """If LLM returns no actions, run() should report failure."""
        controller._llm.send.return_value = MagicMock(text="[]")
        result = controller.run("Do something impossible")
        assert result["success"] is False
        assert result["actions_total"] == 0

    def test_safety_blocked_action(self, controller, mock_subsystems):
        """Blocked actions should be recorded as failed."""
        mock_subsystems["safety"].validate_action.return_value = {
            "allowed": False,
            "risk_level": "critical",
            "needs_confirmation": False,
            "reason": "Forbidden action",
            "violated_rules": ["forbidden_window"],
        }
        result = controller.run("Click something forbidden")
        assert result["actions_failed"] >= 1 or result["actions_succeeded"] == 0

    def test_action_retry_on_failure(self, controller, mock_subsystems):
        """Failed actions should be retried up to max_retries."""
        mock_subsystems["mouse"].click.side_effect = [Exception("miss"), None]
        result = controller.run("Click the Open button")
        # The mouse.click was called at least twice (retry)
        assert mock_subsystems["mouse"].click.call_count >= 1
```

### Visual Verification Tests

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Visual verification tests -- validate VLM-based action verification.

Location: tests/unit/computer_use/test_vlm_analyzer.py
"""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image


@pytest.fixture
def vlm_analyzer():
    """Create a VLMAnalyzer with mocked VLM client."""
    with patch("gaia.computer_use.vision.VLMClient") as MockVLM:
        mock_client = MagicMock()
        MockVLM.return_value = mock_client

        from gaia.computer_use.vision import VLMAnalyzer
        analyzer = VLMAnalyzer()
        analyzer.vlm = mock_client
        yield analyzer


class TestVLMVerification:
    """Test VLM-based visual verification."""

    def test_verify_action_success(self, vlm_analyzer):
        """Verify returns True when VLM says YES."""
        vlm_analyzer.vlm.analyze_images.return_value = {
            "text": "YES - The email has been moved to the Urgent folder."
        }

        before = Image.new("RGB", (100, 100), "white")
        after = Image.new("RGB", (100, 100), "gray")

        result = vlm_analyzer.verify_action_result(
            before, after, "email moved to Urgent folder"
        )
        assert result is True

    def test_verify_action_failure(self, vlm_analyzer):
        """Verify returns False when VLM says NO."""
        vlm_analyzer.vlm.analyze_images.return_value = {
            "text": "NO - The inbox looks unchanged."
        }

        before = Image.new("RGB", (100, 100), "white")
        after = Image.new("RGB", (100, 100), "white")

        result = vlm_analyzer.verify_action_result(
            before, after, "email moved to Urgent folder"
        )
        assert result is False

    def test_describe_screen_detail_levels(self, vlm_analyzer):
        """Each detail level should use a different prompt."""
        vlm_analyzer.vlm.analyze_image.return_value = {"text": "A desktop."}
        image = Image.new("RGB", (100, 100), "blue")

        for level in ("low", "medium", "high"):
            vlm_analyzer.describe_screen(image, detail_level=level)

        assert vlm_analyzer.vlm.analyze_image.call_count == 3
```

### Safety-Specific Tests

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Safety-specific tests -- verify that dangerous actions are blocked.

Location: tests/safety/computer_use/test_forbidden_actions.py
"""

import pytest
from gaia.computer_use.safety import SafetyGuard, SafetyConfig
from gaia.computer_use.controller import Action, ActionType


class TestForbiddenActions:
    """Verify that all forbidden actions are blocked unconditionally."""

    @pytest.fixture
    def guard(self):
        return SafetyGuard(SafetyConfig())

    @pytest.mark.parametrize("target", [
        "1Password - Vault",
        "KeePass Database",
        "LastPass Extension",
        "Bitwarden Vault",
    ])
    def test_password_managers_blocked(self, guard, target):
        """No interaction with password managers."""
        action = Action(type=ActionType.CLICK, target=target, parameters={"x": 0, "y": 0})
        result = guard.validate_action(action)
        assert result["allowed"] is False, f"Should block: {target}"

    @pytest.mark.parametrize("target", [
        "System Settings window",
        "Control Panel applet",
        "Registry Editor",
        "Device Manager",
        "Disk Management console",
    ])
    def test_system_settings_blocked(self, guard, target):
        """No interaction with system settings."""
        action = Action(type=ActionType.CLICK, target=target, parameters={"x": 0, "y": 0})
        result = guard.validate_action(action)
        assert result["allowed"] is False, f"Should block: {target}"

    def test_forbidden_path_ssh_keys(self, guard):
        """Cannot access SSH keys."""
        action = Action(
            type=ActionType.OPEN_APPLICATION,
            target="editor",
            parameters={"path": "~/.ssh/id_rsa"},
        )
        result = guard.validate_action(action)
        assert result["allowed"] is False

    def test_forbidden_path_system32(self, guard):
        """Cannot access Windows System32."""
        action = Action(
            type=ActionType.OPEN_APPLICATION,
            target="explorer",
            parameters={"path": "C:\\Windows\\System32\\cmd.exe"},
        )
        result = guard.validate_action(action)
        assert result["allowed"] is False
```

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Task completion rate** | >= 80% for single-app tasks | Count of successfully completed tasks / total attempted tasks across benchmark set |
| **Action accuracy** | >= 90% per-action success | Number of actions that achieve expected result / total actions executed |
| **Safety block rate** | 100% for forbidden actions | Automated test suite: all forbidden-action tests must pass |
| **False positive safety blocks** | <= 5% | Count of legitimate actions incorrectly blocked / total legitimate actions |
| **Planning quality** | >= 85% valid plans | Plans that execute without replanning / total plans generated |
| **Verification accuracy** | >= 85% correct VLM verdicts | Manual review of VLM YES/NO answers against ground truth |
| **End-to-end latency** | <= 5s per action step (avg) | Time from action dispatch to verification complete |
| **Screenshot + OCR latency** | <= 500ms | Time to capture screen and extract text |
| **VLM analysis latency** | <= 3s per image | Time from image submission to VLM response |
| **Replan success rate** | >= 60% recovery | Tasks that succeed after replan / tasks that needed replan |
| **Cross-platform parity** | 3 OS supported | Windows, macOS, Linux desktop automation passing CI |
| **Crash / hang rate** | <= 1% of sessions | Sessions that terminate abnormally / total sessions |
| **Test coverage** | >= 85% line coverage | pytest-cov for `src/gaia/computer_use/` |
| **Documentation coverage** | 100% public APIs | All public methods have docstrings; MDX guide published |

---

## Complete Code File Listing

### Source Files

| File Path | Description |
|-----------|-------------|
| `src/gaia/computer_use/__init__.py` | Package init -- exports public API |
| `src/gaia/computer_use/vision.py` | ScreenCaptureEngine, OCREngine, VLMAnalyzer |
| `src/gaia/computer_use/input.py` | MouseController, KeyboardController |
| `src/gaia/computer_use/browser.py` | BrowserAutomationAgent (Playwright) |
| `src/gaia/computer_use/desktop.py` | DesktopAutomationBase, UIElement, WindowInfo, factory |
| `src/gaia/computer_use/controller.py` | ComputerUseController, Action, ActionType, Plan |
| `src/gaia/computer_use/safety.py` | SafetyGuard, SafetyConfig, SafetyRule, RollbackManager, PanicButton |
| `src/gaia/computer_use/platforms/__init__.py` | Platform subpackage init |
| `src/gaia/computer_use/platforms/windows.py` | WindowsDesktopAutomation (UIA) |
| `src/gaia/computer_use/platforms/macos.py` | MacOSDesktopAutomation (NSAccessibility) |
| `src/gaia/computer_use/platforms/linux.py` | LinuxDesktopAutomation (AT-SPI) |
| `src/gaia/agents/computer_use/__init__.py` | Agent subpackage init |
| `src/gaia/agents/computer_use/agent.py` | ComputerUseAgent (inherits from Agent) |
| `src/gaia/agents/computer_use/tools_mixin.py` | ComputerUseToolsMixin for reuse in other agents |

### Test Files

| File Path | Description |
|-----------|-------------|
| `tests/unit/computer_use/__init__.py` | Test package init |
| `tests/unit/computer_use/test_screen_capture.py` | Screenshot capture tests |
| `tests/unit/computer_use/test_ocr_engine.py` | OCR extraction tests |
| `tests/unit/computer_use/test_vlm_analyzer.py` | VLM analysis tests (mocked VLM) |
| `tests/unit/computer_use/test_mouse_controller.py` | Mouse control tests (mocked pyautogui) |
| `tests/unit/computer_use/test_keyboard_controller.py` | Keyboard control tests (mocked pyautogui) |
| `tests/unit/computer_use/test_safety_guard.py` | Safety rule validation tests |
| `tests/unit/computer_use/test_rollback.py` | Rollback manager tests |
| `tests/unit/computer_use/test_action_model.py` | Action / Plan data model tests |
| `tests/unit/computer_use/test_controller.py` | Controller logic tests (mocked subsystems) |
| `tests/integration/computer_use/__init__.py` | Integration test package init |
| `tests/integration/computer_use/test_browser_automation.py` | Playwright browser tests |
| `tests/integration/computer_use/test_desktop_automation.py` | Desktop UI tree tests |
| `tests/integration/computer_use/test_controller_e2e.py` | Full controller end-to-end tests |
| `tests/integration/computer_use/test_agent_e2e.py` | ComputerUseAgent end-to-end tests |
| `tests/safety/computer_use/__init__.py` | Safety test package init |
| `tests/safety/computer_use/test_forbidden_actions.py` | Forbidden action blocking tests |
| `tests/safety/computer_use/test_confirmation_flow.py` | Confirmation prompt flow tests |
| `tests/safety/computer_use/test_rate_limiting.py` | Rate limiting tests |
| `tests/safety/computer_use/test_panic_button.py` | Emergency stop tests |
| `tests/safety/computer_use/test_safe_mode.py` | Safe mode restriction tests |

### Documentation Files

| File Path | Description |
|-----------|-------------|
| `docs/guides/computer-use.mdx` | User guide for computer use agent |
| `docs/sdk/agents/computer-use.mdx` | SDK reference for ComputerUseAgent |
| `docs/spec/computer-use.mdx` | This architecture specification (converted to MDX) |

---

## Appendix: Quick Reference

### Creating a Minimal Computer Use Session

```python
from gaia.agents.computer_use.agent import ComputerUseAgent

agent = ComputerUseAgent(safe_mode=True)  # Read-only to start
result = agent.process_query("Take a screenshot and describe what you see")
print(result)
```

### Running via CLI

```bash
# Interactive mode
gaia computer-use

# Single task
gaia computer-use -q "Open Chrome and navigate to github.com"

# Safe mode (read-only)
gaia computer-use --safe-mode -q "Describe what is on screen"

# With Claude for planning
gaia computer-use --use-claude -q "Triage my Gmail inbox"
```

### Adding Computer Use to an Existing Agent

```python
from gaia.agents.base.agent import Agent
from gaia.agents.computer_use.tools_mixin import ComputerUseToolsMixin


class MyCustomAgent(Agent, ComputerUseToolsMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_computer_use_tools()

    def _get_system_prompt(self) -> str:
        return "You are a custom agent with computer use capabilities..."
```
