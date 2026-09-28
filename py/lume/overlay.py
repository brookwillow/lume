"""Floating HUD overlay for Lume voice assistant.

A borderless, always-on-top, click-through window that shows:
- Recording state (pulsing indicator)
- ASR transcription result
- Agent response
"""

import objc
import math
import time as _time
import unicodedata
from AppKit import (
    NSWindow,
    NSWindowStyleMaskBorderless,
    NSBackingStoreBuffered,
    NSColor,
    NSTextField,
    NSFont,
    NSScreen,
    NSView,
    NSFloatingWindowLevel,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorStationary,
    NSLineBreakByWordWrapping,
    NSTextAlignmentLeft,
    NSTextAlignmentCenter,
    NSAnimationContext,
    NSVisualEffectView,
    NSVisualEffectMaterialHUDWindow,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectStateActive,
    NSImage,
    NSBezierPath,
    NSCompositingOperationSourceOver,
)
from Foundation import NSMakeRect, NSMakeSize, NSTimer, NSObject
from Quartz import CAMediaTimingFunction


# Window dimensions
_WIDTH_COMPACT = 220    # Waveform / processing
_WIDTH_TEXT = 320       # Preferred text width
_WIDTH_TEXT_MAX = 420   # Grow only when a short status benefits from staying on one line
_HEIGHT_COMPACT = 36    # Single-line / waveform
_HEIGHT_RESULT = 52     # Multi-line result
_HEIGHT_MAX = 420       # Maximum height for wrapped text
# NSTextField's 10pt system font needs a 16px control frame to keep its
# baseline optically centered. A 14px frame clips its natural ascent/descent.
_LINE_HEIGHT = 16
_MARGIN_BOTTOM = 100
_CORNER_RADIUS = 10
_PADDING_H = 10
_PADDING_V = 8
_INDICATOR_WIDTH = 18


def _rounded_rect_mask(width, height, radius):
    """Create an NSImage mask with rounded rect for NSVisualEffectView."""
    size = NSMakeSize(width, height)
    image = NSImage.alloc().initWithSize_(size)
    image.lockFocus()
    path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
        NSMakeRect(0, 0, width, height), radius, radius
    )
    NSColor.blackColor().setFill()
    path.fill()
    image.unlockFocus()
    image.setCapInsets_((radius, radius, radius, radius))
    return image


# ─── Waveform Animation View ───

_NUM_BARS = 36
_BAR_WIDTH = 1.8
_BAR_GAP = 0.8
_BAR_RADIUS = 0.9
_NOISE_GATE = 0.008      # Below this level, bars stay flat (noise suppression)
_SMOOTH_ATTACK = 0.7     # How fast level rises (0-1, higher=faster)
_SMOOTH_RELEASE = 0.15   # How fast level drops (0-1, lower=slower fade)


def _estimated_text_width(text):
    """Estimate 10pt HUD text width across Latin and CJK content."""
    width = 0.0
    for char in text:
        width += 10.0 if unicodedata.east_asian_width(char) in ("F", "W") else 5.5
    return width


def _estimated_line_count(text, width):
    """Estimate wrapped line count for the HUD label."""
    if width <= 0:
        return 1
    lines = 0
    for segment in str(text).splitlines() or [""]:
        lines += max(1, int(math.ceil(_estimated_text_width(segment) / width)))
    return lines


class WaveformView(NSView):
    """Custom NSView that draws sound wave bars driven by real audio level."""

    def initWithFrame_(self, frame):
        self = objc.super(WaveformView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._animating = False
        self._timer = None
        self._start_time = 0
        self._level = 0.0          # Current smoothed audio level (0-1)
        self._target_level = 0.0   # Raw target level from audio
        self._bar_heights = [0.0] * _NUM_BARS  # Per-bar smoothed heights
        return self

    def setAudioLevel_(self, level):
        """Update audio level from recording stream. level: 0.0-1.0."""
        # Apply noise gate
        if level < _NOISE_GATE:
            self._target_level = 0.0
        else:
            # Normalize above gate, then apply power curve to amplify speech
            gated = (level - _NOISE_GATE) / (1.0 - _NOISE_GATE)
            # Square root expands low-mid range, making speech more visible
            self._target_level = min(1.0, gated ** 0.4)

    def drawRect_(self, rect):
        if not self._animating:
            return

        bounds = self.bounds()
        w = bounds.size.width
        h = bounds.size.height
        t = _time.time() - self._start_time

        # Smooth the level (fast attack, slow release)
        if self._target_level > self._level:
            self._level += (self._target_level - self._level) * _SMOOTH_ATTACK
        else:
            self._level += (self._target_level - self._level) * _SMOOTH_RELEASE

        total_bar_width = _NUM_BARS * _BAR_WIDTH + (_NUM_BARS - 1) * _BAR_GAP
        x_offset = (w - total_bar_width) / 2
        center = _NUM_BARS / 2.0

        for i in range(_NUM_BARS):
            # Distance from center (0.0 at center, 1.0 at edges)
            dist = abs(i - center) / center
            # Smooth fade envelope (no hard cutoff at edges)
            envelope = math.cos(dist * math.pi * 0.5) ** 0.6

            # Subtle wave motion (only visible when speaking)
            phase = i * 0.25
            wave = (
                math.sin(t * 8.0 + phase) * 0.15
                + math.sin(t * 12.0 + phase * 1.3) * 0.1
            )

            # Mix: when level is 0, bars are minimal; when speaking, bars animate
            speech_amp = self._level * (0.7 + wave) * envelope
            idle_amp = 0.04 * envelope  # Tiny idle bars
            amplitude = idle_amp + speech_amp

            bar_h = max(2, amplitude * h * 0.9)

            x = x_offset + i * (_BAR_WIDTH + _BAR_GAP)
            y = (h - bar_h) / 2

            # Color: brighter when speaking
            hue = 0.55 + dist * 0.12
            brightness = 0.5 + self._level * 0.5
            alpha = 0.3 + self._level * 0.6 + envelope * 0.1
            color = NSColor.colorWithCalibratedHue_saturation_brightness_alpha_(
                hue, 0.7, brightness, alpha
            )
            color.setFill()

            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x, y, _BAR_WIDTH, bar_h), _BAR_RADIUS, _BAR_RADIUS
            )
            path.fill()

    def startAnimating(self):
        self._animating = True
        self._start_time = _time.time()
        self._level = 0.0
        self._target_level = 0.0
        self.setHidden_(False)
        # 45fps for responsive level-driven animation
        self._timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            1.0 / 45.0, True, lambda _: self.setNeedsDisplay_(True)
        )

    def stopAnimating(self):
        self._animating = False
        self._level = 0.0
        self._target_level = 0.0
        if self._timer:
            self._timer.invalidate()
            self._timer = None
        self.setHidden_(True)


class LumeOverlay:
    """Global floating HUD overlay."""

    def __init__(self):
        self._window = None
        self._effect_view = None
        self._dot_view = None
        self._label = None
        self._sublabel = None
        self._waveform = None
        self._hide_timer = None
        self._cur_width = _WIDTH_COMPACT
        self._cur_height = _HEIGHT_COMPACT
        self._setup()

    def _setup(self):
        frame = NSMakeRect(0, 0, _WIDTH_COMPACT, _HEIGHT_COMPACT)

        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )
        self._window.setLevel_(NSFloatingWindowLevel + 2)
        self._window.setOpaque_(False)
        self._window.setBackgroundColor_(NSColor.clearColor())
        self._window.setHasShadow_(True)
        self._window.setIgnoresMouseEvents_(True)
        self._window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
        )
        self._window.setAlphaValue_(0.0)

        # Frosted glass content view with layer-based rounding & border
        content_frame = NSMakeRect(0, 0, _WIDTH_COMPACT, _HEIGHT_COMPACT)
        self._effect_view = NSVisualEffectView.alloc().initWithFrame_(content_frame)
        self._effect_view.setMaterial_(NSVisualEffectMaterialHUDWindow)
        self._effect_view.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        self._effect_view.setState_(NSVisualEffectStateActive)
        self._effect_view.setWantsLayer_(True)
        self._effect_view.layer().setCornerRadius_(_CORNER_RADIUS)
        self._effect_view.layer().setMasksToBounds_(True)
        self._effect_view.layer().setBorderWidth_(1.5)
        self._effect_view.layer().setBorderColor_(
            NSColor.colorWithCalibratedWhite_alpha_(0.45, 0.7).CGColor()
        )
        self._window.setContentView_(self._effect_view)

        # Dark overlay background
        self._bg_view = NSView.alloc().initWithFrame_(content_frame)
        self._bg_view.setWantsLayer_(True)
        self._bg_view.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.08, 0.08, 0.10, 0.85).CGColor()
        )
        self._effect_view.addSubview_(self._bg_view)

        # Indicator glyph for state-specific symbols. It has a fixed line-height
        # frame so multi-line labels never stretch or squash the glyph.
        self._dot_view = self._make_label(
            NSMakeRect(_PADDING_H, 0, _INDICATOR_WIDTH, _HEIGHT_COMPACT),
            font_size=10,
            color=NSColor.colorWithCalibratedWhite_alpha_(0.5, 1.0),
        )
        self._dot_view.setAlignment_(NSTextAlignmentCenter)
        self._dot_view.setMaximumNumberOfLines_(1)
        self._effect_view.addSubview_(self._dot_view)

        # Main label
        self._label = self._make_label(
            NSMakeRect(_PADDING_H + _INDICATOR_WIDTH + 6, 0, _WIDTH_TEXT - _PADDING_H * 2 - _INDICATOR_WIDTH - 6, _HEIGHT_COMPACT),
            font_size=10,
            bold=True,
            color=NSColor.colorWithCalibratedWhite_alpha_(0.82, 1.0),
        )
        self._label.setMaximumNumberOfLines_(0)
        self._effect_view.addSubview_(self._label)

        # Sub-label
        self._sublabel = self._make_label(
            NSMakeRect(_PADDING_H + 14, 0, _WIDTH_TEXT - _PADDING_H * 2 - 14, 10),
            font_size=9,
            color=NSColor.colorWithCalibratedWhite_alpha_(0.55, 1.0),
        )
        self._sublabel.setMaximumNumberOfLines_(0)
        self._effect_view.addSubview_(self._sublabel)

        # Waveform animation view
        self._waveform = WaveformView.alloc().initWithFrame_(
            NSMakeRect(0, 0, _WIDTH_COMPACT, _HEIGHT_COMPACT)
        )
        self._waveform.setHidden_(True)
        self._effect_view.addSubview_(self._waveform)

    def _make_label(self, frame, font_size=14, bold=False, color=None):
        label = NSTextField.alloc().initWithFrame_(frame)
        label.setStringValue_("")
        if bold:
            label.setFont_(NSFont.systemFontOfSize_weight_(font_size, 0.56))
        else:
            label.setFont_(NSFont.systemFontOfSize_(font_size))
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setTextColor_(color or NSColor.labelColor())
        label.setLineBreakMode_(NSLineBreakByWordWrapping)
        label.setAlignment_(NSTextAlignmentLeft)
        cell = label.cell()
        if cell:
            cell.setWraps_(True)
            cell.setScrollable_(False)
            cell.setUsesSingleLineMode_(False)
            cell.setLineBreakMode_(NSLineBreakByWordWrapping)
        return label

    def _resize_window(self, width, height):
        """Resize overlay window and reposition content."""
        frame = self._window.frame()
        actual_matches = (
            abs(frame.size.width - width) < 0.5
            and abs(frame.size.height - height) < 0.5
        )
        if width == self._cur_width and height == self._cur_height and actual_matches:
            return
        self._cur_width = width
        self._cur_height = height

        # Update window frame (keep centered on current position)
        screen = NSScreen.mainScreen()
        if screen:
            sf = screen.frame()
            x = sf.origin.x + (sf.size.width - width) / 2
            y = sf.origin.y + _MARGIN_BOTTOM
            self._window.setFrame_display_(NSMakeRect(x, y, width, height), True)

        # Update effect view
        self._effect_view.setFrame_(NSMakeRect(0, 0, width, height))

        # Update background view
        self._bg_view.setFrame_(NSMakeRect(0, 0, width, height))

        # Update waveform size
        self._waveform.setFrame_(NSMakeRect(0, 0, width, height))

    def _single_line_width_for_text(self, text):
        """Return the smallest HUD width that can comfortably fit one line."""
        # Keep equal padding on both sides so dynamically-sized single-line
        # HUDs do not appear to drift horizontally.
        content_w = _PADDING_H * 2 + _INDICATOR_WIDTH + 6 + _estimated_text_width(text)
        return min(_WIDTH_TEXT_MAX, max(_WIDTH_COMPACT, int(math.ceil(content_w))))

    def _width_for_status(self, title, detail=""):
        """Keep short status text compact while giving longer text room to breathe."""
        detail_lines = str(detail).splitlines() or [""]
        widest = max([str(title), *detail_lines], key=_estimated_text_width)
        return self._single_line_width_for_text(widest)

    def _height_for_label_text(self, text, width):
        """Measure wrapped NSTextField content height using AppKit's text cell."""
        return self._height_for_text_field(self._label, text, width, _LINE_HEIGHT)

    def _height_for_text_field(self, field, text, width, line_height):
        """Measure a label without relying on a stale layout frame."""
        field.setFrame_(NSMakeRect(0, 0, width, _HEIGHT_MAX))
        field.setStringValue_(text)
        cell = field.cell()
        estimated = _estimated_line_count(text, width) * line_height
        if cell:
            measured = cell.cellSizeForBounds_(NSMakeRect(0, 0, width, _HEIGHT_MAX)).height
            return max(line_height, measured, estimated)
        return estimated

    def _layout_text(self, has_sublabel=False, multiline=False, main_height=None, detail_height=None):
        """Layout text labels within current window size."""
        w = self._cur_width
        h = self._cur_height
        label_x = _PADDING_H + _INDICATOR_WIDTH + 6
        label_w = w - _PADDING_H * 2 - _INDICATOR_WIDTH - 6

        if has_sublabel:
            main_height = main_height or 16
            detail_height = detail_height or 14
            label_frame = NSMakeRect(
                label_x, h - _PADDING_V - main_height, label_w, main_height
            )
            self._label.setFrame_(label_frame)
            self._sublabel.setFrame_(
                NSMakeRect(label_x, _PADDING_V, label_w, detail_height)
            )
        elif multiline:
            label_frame = NSMakeRect(label_x, _PADDING_V - 1, label_w, h - _PADDING_V * 2)
            self._label.setFrame_(label_frame)
        else:
            label_frame = NSMakeRect(
                label_x, (h - _LINE_HEIGHT) / 2, label_w, _LINE_HEIGHT
            )
            self._label.setFrame_(label_frame)

        self._dot_view.setFrame_(
            NSMakeRect(
                _PADDING_H, (h - _LINE_HEIGHT) / 2,
                _INDICATOR_WIDTH, _LINE_HEIGHT,
            )
        )

    def _show_activity(self, symbol, color, title, detail=""):
        """Render the compact execution view: current state plus one recent step."""
        self._cancel_hide()
        self._stop_dots_animation()
        self._waveform.stopAnimating()
        self._dot_view.setHidden_(False)
        self._label.setHidden_(False)
        self._sublabel.setHidden_(not bool(detail))
        self._set_indicator(symbol, color)

        title = str(title)
        detail = str(detail)
        width = self._width_for_status(title, detail)
        label_w = width - _PADDING_H * 2 - _INDICATOR_WIDTH - 6
        main_lines = _estimated_line_count(title, label_w)
        # AppKit occasionally measures a single line a few pixels taller than
        # its line height. Use a fixed frame for true single-line states so
        # their baseline stays centered with the indicator.
        if main_lines == 1:
            main_h = _LINE_HEIGHT
        else:
            main_h = min(self._height_for_label_text(title, label_w), _LINE_HEIGHT * 2)
        self._label.setMaximumNumberOfLines_(max(1, int(math.ceil(main_h / _LINE_HEIGHT))))
        self._label.setStringValue_(title)
        self._sublabel.setStringValue_(detail)

        if detail:
            detail_h = min(
                self._height_for_text_field(self._sublabel, detail, label_w, 14),
                _LINE_HEIGHT * 4,
            )
            self._sublabel.setMaximumNumberOfLines_(max(1, int(math.ceil(detail_h / 14))))
            height = int(math.ceil(_PADDING_V * 2 + main_h + detail_h + 4))
        elif main_h <= _LINE_HEIGHT + 2:
            detail_h = 0
            height = _HEIGHT_COMPACT
        else:
            detail_h = 0
            height = int(math.ceil(_PADDING_V * 2 + main_h + 2))
        height = min(_HEIGHT_MAX, height)
        self._resize_window(width, height)
        self._layout_text(
            has_sublabel=bool(detail),
            multiline=main_lines > 1,
            main_height=main_h,
            detail_height=detail_h,
        )
        self._ensure_visible()

    # ─── Public API (call from main thread via AppHelper.callAfter) ───

    def _set_indicator(self, symbol, color):
        """Set indicator glyph and color."""
        self._dot_view.setStringValue_(symbol)
        self._dot_view.setTextColor_(color)

    def update_audio_level(self, level: float):
        """Feed real-time audio level to waveform. level: 0.0-1.0."""
        self._waveform.setAudioLevel_(level)

    def show_listening(self):
        """Show recording state with waveform animation."""
        self._cancel_hide()
        self._stop_dots_animation()
        self._waveform.stopAnimating()
        self._label.setStringValue_("")
        self._sublabel.setStringValue_("")
        self._resize_window(_WIDTH_COMPACT, _HEIGHT_COMPACT)
        self._dot_view.setHidden_(True)
        self._label.setHidden_(True)
        self._sublabel.setHidden_(True)
        self._waveform.startAnimating()
        self._fade_in()

    def show_processing(self):
        """Show ASR processing state with animated dots."""
        self._waveform.stopAnimating()
        self._resize_window(_WIDTH_COMPACT, _HEIGHT_COMPACT)
        self._dot_view.setHidden_(False)
        self._label.setHidden_(False)
        self._sublabel.setHidden_(True)
        self._set_indicator("◎", NSColor.colorWithCalibratedWhite_alpha_(0.5, 1.0))
        self._label.setMaximumNumberOfLines_(1)
        self._label.setStringValue_("···")
        self._layout_text(has_sublabel=False)
        self._start_dots_animation()

    def show_thinking(self, detail=""):
        """Show a semantic reasoning state without exposing model internals."""
        self._show_activity(
            "◌",
            NSColor.colorWithCalibratedWhite_alpha_(0.65, 1.0),
            "正在判断下一步",
            detail,
        )

    def show_action(self, title, previous_step=""):
        """Show the action in progress and, at most, the immediately prior step."""
        self._show_activity(
            "◐",
            NSColor.colorWithCalibratedWhite_alpha_(0.78, 1.0),
            title,
            previous_step,
        )

    def show_action_result(self, title, status="unknown", previous_step=""):
        """Show the observable tool outcome before the agent chooses its next step."""
        if status == "success":
            symbol = "✓"
            color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.35, 0.88, 0.62, 1.0)
            prefix = "已完成："
        elif status == "failed":
            symbol = "×"
            color = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.43, 0.40, 1.0)
            prefix = "未完成："
        else:
            symbol = "!"
            color = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.72, 0.30, 1.0)
            prefix = "已尝试："
        self._show_activity(symbol, color, prefix + title, previous_step)

    def show_transcription(self, text):
        """Show ASR result."""
        self._show_activity(
            "◉",
            NSColor.colorWithCalibratedWhite_alpha_(0.70, 1.0),
            str(text),
        )

    def show_result(self, text, hide_after=5.0):
        """Show agent result and auto-hide."""
        self._stop_dots_animation()
        self._waveform.stopAnimating()
        self._dot_view.setHidden_(False)
        self._label.setHidden_(False)
        self._sublabel.setHidden_(True)
        display = str(text)
        if any(marker in display for marker in ("已尝试", "无法确认", "未能", "失败", "错误")):
            self._set_indicator(
                "!", NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.72, 0.30, 1.0)
            )
        else:
            self._set_indicator(
                "✓", NSColor.colorWithCalibratedRed_green_blue_alpha_(0.35, 0.88, 0.62, 1.0)
            )

        # Auto-size: let the full result wrap and expand window height as needed.
        self._label.setStringValue_(display)
        width = self._width_for_status(display)
        label_w = width - _PADDING_H * 2 - _INDICATOR_WIDTH - 6
        line_count = _estimated_line_count(display, label_w)
        if line_count == 1:
            label_h = _LINE_HEIGHT
        else:
            measured_label_h = self._height_for_label_text(display, label_w)
            label_h = min(measured_label_h, _HEIGHT_MAX - _PADDING_V * 2)
        self._label.setMaximumNumberOfLines_(
            max(1, int(math.ceil(label_h / _LINE_HEIGHT)))
        )

        if label_h <= _LINE_HEIGHT + 2:
            height = _HEIGHT_COMPACT
        else:
            height = min(_HEIGHT_MAX, int(math.ceil(_PADDING_V * 2 + label_h + 2)))

        self._resize_window(width, height)
        self._layout_text(multiline=(height > _HEIGHT_COMPACT))

        self._ensure_visible()
        if hide_after is not None:
            self._schedule_hide(hide_after)

    def show_error(self, text):
        """Show error and auto-hide."""
        self._stop_dots_animation()
        self._waveform.stopAnimating()
        self._dot_view.setHidden_(False)
        self._label.setHidden_(False)
        self._sublabel.setHidden_(True)
        self._set_indicator("○", NSColor.colorWithCalibratedWhite_alpha_(0.4, 1.0))
        self._label.setMaximumNumberOfLines_(0)
        display = str(text)
        self._label.setStringValue_(display)
        width = self._width_for_status(display)
        label_w = width - _PADDING_H * 2 - _INDICATOR_WIDTH - 6
        line_count = _estimated_line_count(display, label_w)
        if line_count == 1:
            label_h = _LINE_HEIGHT
        else:
            label_h = min(
                self._height_for_label_text(display, label_w),
                _HEIGHT_MAX - _PADDING_V * 2,
            )
        self._label.setMaximumNumberOfLines_(max(1, int(math.ceil(label_h / _LINE_HEIGHT))))
        height = _HEIGHT_COMPACT if label_h <= _LINE_HEIGHT + 2 else int(math.ceil(_PADDING_V * 2 + label_h + 2))
        self._resize_window(width, min(_HEIGHT_MAX, height))
        self._layout_text(multiline=(height > _HEIGHT_COMPACT))
        self._schedule_hide(3.0)

    def hide(self):
        """Fade out and hide."""
        self._fade_out()

    # ─── Private ───

    def _move_to_active_screen(self):
        """Position the overlay on the currently focused screen, horizontally centered."""
        screen = NSScreen.mainScreen()
        if screen is None:
            return
        sf = screen.frame()
        x = sf.origin.x + (sf.size.width - self._cur_width) / 2
        y = sf.origin.y + _MARGIN_BOTTOM
        self._window.setFrameOrigin_((x, y))

    def _ensure_visible(self):
        """Make sure HUD is visible without replaying the full fade-in animation."""
        if self._window.alphaValue() > 0.5:
            # Already visible — just ensure correct position/size
            self._move_to_active_screen()
            return
        # Not visible yet — do a full fade in
        self._fade_in()

    def _fade_in(self):
        self._move_to_active_screen()
        self._window.orderFrontRegardless()
        # Only animate opacity. Scaling the NSWindow frame during fade-out left
        # the real frame smaller than _cur_width/_cur_height, so later states
        # could compound the shrink on every desktop switch.
        if self._window.alphaValue() > 0.5:
            self._window.setAlphaValue_(1.0)
            return
        self._window.setAlphaValue_(0.0)
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.25)
        NSAnimationContext.currentContext().setTimingFunction_(
            CAMediaTimingFunction.functionWithName_('easeOut')
        )
        self._window.animator().setAlphaValue_(1.0)
        NSAnimationContext.endGrouping()

    def _fade_out(self):
        # Keep the window frame intact; only its opacity should change.
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.2)
        NSAnimationContext.currentContext().setTimingFunction_(
            CAMediaTimingFunction.functionWithName_('easeIn')
        )
        self._window.animator().setAlphaValue_(0.0)
        NSAnimationContext.endGrouping()

    def _schedule_hide(self, delay):
        self._cancel_hide()
        self._hide_timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            delay, False, lambda _: self.hide()
        )

    def _cancel_hide(self):
        if self._hide_timer:
            self._hide_timer.invalidate()
            self._hide_timer = None

    def _start_dots_animation(self):
        """Animate dots: · → ·· → ··· → · ..."""
        self._stop_dots_animation()
        self._dots_count = 1

        def _tick(_):
            self._dots_count = (self._dots_count % 3) + 1
            self._label.setStringValue_("·" * self._dots_count)

        self._dots_timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            0.4, True, _tick
        )

    def _stop_dots_animation(self):
        timer = getattr(self, "_dots_timer", None)
        if timer:
            timer.invalidate()
            self._dots_timer = None
