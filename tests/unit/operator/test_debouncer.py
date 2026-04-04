"""Unit tests for the template rendering debouncer."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tests.unit.conftest import (
    patch_debouncer_environment,
    patch_debouncer_logger,
    create_debouncer_mock_factory,
    managed_debouncer,
)
from haproxy_template_ic.models.templates import TriggerContext

# Test constants to avoid magic numbers
MIN_INTERVAL_SHORT = 0.04  # 40ms for fast test execution
MAX_INTERVAL_SHORT = 0.12  # 120ms for guaranteed execution tests
MIN_INTERVAL_LONG = 0.2  # 200ms for rate limiting tests
MAX_INTERVAL_LONG = 5.0  # 5s to not interfere with rate limiting tests
SLEEP_BUFFER = 0.02  # 20ms buffer for async operations
SLEEP_SHORT = 0.08  # 80ms wait after trigger
SLEEP_MEDIUM = 0.15  # 150ms to exceed short max_interval
SLEEP_LONG = 0.25  # 250ms to ensure render completes


# =============================================================================
# Centralized Mock Creation
# =============================================================================


# Use the centralized debouncer factory from conftest.py
def create_debouncer(
    min_interval=MIN_INTERVAL_SHORT, max_interval=MAX_INTERVAL_LONG, **kwargs
):
    """Create a TemplateRenderDebouncer with standard mocks."""
    return create_debouncer_mock_factory(
        min_interval=min_interval, max_interval=max_interval, **kwargs
    )


@pytest.fixture
def mock_render_haproxy_templates():
    """Mock the render_haproxy_templates function."""
    with patch(
        "haproxy_template_ic.operator.debouncer.render_haproxy_templates",
        new_callable=AsyncMock,
    ) as mock:
        # Make the mock return successfully without doing anything
        mock.return_value = None
        yield mock


@pytest.mark.asyncio
async def test_init_validation():
    """Test that initialization validates interval constraints."""
    # Valid initialization
    debouncer = create_debouncer(min_interval=5, max_interval=10)
    assert debouncer.min_interval == 5
    assert debouncer.max_interval == 10

    # Invalid: max < min
    with pytest.raises(ValueError, match="max_interval.*must be >= min_interval"):
        create_debouncer(min_interval=10, max_interval=5)


@pytest.mark.asyncio
async def test_single_trigger(mock_render_haproxy_templates):
    """Test that a single trigger causes rendering after min_interval."""
    async with managed_debouncer(MIN_INTERVAL_SHORT, MAX_INTERVAL_LONG) as debouncer:
        # Trigger once
        await debouncer.trigger()

        # Wait for min_interval + buffer
        await asyncio.sleep(SLEEP_SHORT)

        # Should have rendered once
        assert mock_render_haproxy_templates.call_count == 1

        # Verify the call was made with correct arguments
        call_args = mock_render_haproxy_templates.call_args
        assert call_args is not None
        args, kwargs = call_args

        # Check that trigger_context is passed in kwargs
        assert "trigger_context" in kwargs
        trigger_context = kwargs["trigger_context"]
        assert isinstance(trigger_context, TriggerContext)
        assert trigger_context.pod_changed is False


@pytest.mark.asyncio
async def test_guaranteed_periodic_execution(mock_render_haproxy_templates):
    """Test that rendering happens at max_interval even without triggers."""
    async with managed_debouncer(MIN_INTERVAL_SHORT, MAX_INTERVAL_SHORT) as _debouncer:
        # Don't trigger anything, just wait
        await asyncio.sleep(SLEEP_MEDIUM)  # Wait past max_interval

        # Should have rendered due to timeout
        assert mock_render_haproxy_templates.call_count >= 1


@pytest.mark.asyncio
async def test_rate_limiting(mock_render_haproxy_templates):
    """Test that min_interval is enforced between renders."""

    async with managed_debouncer(MIN_INTERVAL_LONG, MAX_INTERVAL_LONG) as debouncer:
        await asyncio.sleep(MIN_INTERVAL_SHORT)  # Let it start

        # First trigger and render
        await debouncer.trigger()
        await asyncio.sleep(SLEEP_LONG)  # Wait for render
        render_count_1 = mock_render_haproxy_templates.call_count
        assert render_count_1 >= 1

        # Trigger again - should be rate limited
        await debouncer.trigger()
        await asyncio.sleep(SLEEP_SHORT)  # Less than min_interval

        # Should not have rendered yet
        assert mock_render_haproxy_templates.call_count == render_count_1

        # Wait for min_interval to pass
        await asyncio.sleep(SLEEP_SHORT * 2)

        # Now should have rendered again
        assert mock_render_haproxy_templates.call_count > render_count_1


@pytest.mark.asyncio
async def test_stop_gracefully(mock_render_haproxy_templates):
    """Test that debouncer stops gracefully."""
    debouncer = create_debouncer(MIN_INTERVAL_SHORT, MAX_INTERVAL_LONG)

    await debouncer.start()
    await asyncio.sleep(SLEEP_BUFFER)  # Let it start

    # Stop should complete without errors
    await debouncer.stop()

    # Task should be done
    assert debouncer._task.done()
    assert debouncer._stop is True


@pytest.mark.asyncio
async def test_render_error_handling(mock_render_haproxy_templates):
    """Test that render errors don't crash the debouncer."""
    # Configure the mock to raise an exception using helper
    mock_render_haproxy_templates.side_effect = Exception("Test error")

    async with managed_debouncer(MIN_INTERVAL_SHORT, MIN_INTERVAL_LONG) as debouncer:
        # Trigger rendering
        await debouncer.trigger()
        await asyncio.sleep(SLEEP_LONG)

        # Should have attempted render despite error
        assert mock_render_haproxy_templates.call_count >= 1


@pytest.mark.asyncio
async def test_start_idempotent(mock_render_haproxy_templates):
    """Test that calling start multiple times is safe."""

    debouncer = create_debouncer(
        min_interval=MIN_INTERVAL_SHORT,
        max_interval=MAX_INTERVAL_LONG,
    )

    await debouncer.start()
    first_task = debouncer._task

    # Starting again should not create a new task
    with patch_debouncer_environment() as mock_logger:
        await debouncer.start()
        mock_logger.warning.assert_called_with("Debouncer already running")

    assert debouncer._task is first_task

    await debouncer.stop()


@pytest.mark.asyncio
async def test_periodic_refresh_timing(mock_render_haproxy_templates):
    """Test that periodic refresh doesn't add unnecessary min_interval wait."""
    import time

    async with managed_debouncer(
        min_interval=MIN_INTERVAL_SHORT,  # 0.1s min interval
        max_interval=MAX_INTERVAL_SHORT,  # 0.3s max interval (short for test)
    ) as _debouncer:
        start_time = time.time()

        # Wait for periodic refresh (should happen after max_interval)
        await asyncio.sleep(SLEEP_MEDIUM)  # 0.35s > 0.3s max_interval

        # Should have rendered due to periodic refresh
        assert mock_render_haproxy_templates.call_count >= 1

        # Check that total time is approximately max_interval, not max_interval + min_interval
        total_time = time.time() - start_time
        # Allow some buffer for timing precision, but should be much closer to max_interval (0.3s)
        # than max_interval + min_interval (0.3s + 0.1s = 0.4s)
        assert total_time < 0.4  # Should be much less than 0.4s if working correctly


@pytest.mark.asyncio
async def test_accurate_time_calculation_after_wait(mock_render_haproxy_templates):
    """Test that resource changes don't wait unnecessarily when enough time has passed."""
    import time

    async with managed_debouncer(
        min_interval=MIN_INTERVAL_LONG,  # 0.5s min interval
        max_interval=MAX_INTERVAL_LONG,  # 10s max interval
    ) as debouncer:
        # First render
        await debouncer.trigger()
        await asyncio.sleep(SLEEP_LONG)  # 0.6s wait for first render
        first_count = mock_render_haproxy_templates.call_count
        assert first_count >= 1

        # Wait longer than min_interval
        await asyncio.sleep(SLEEP_LONG)  # Another 0.6s, total 1.2s > 0.5s min_interval

        start_time = time.time()
        # Second trigger should not need additional min_interval wait
        await debouncer.trigger()
        await asyncio.sleep(SLEEP_SHORT)  # Just enough for processing

        # Should have rendered without additional delay
        assert mock_render_haproxy_templates.call_count > first_count
        processing_time = time.time() - start_time
        # Should be much less than min_interval (0.5s) since enough time already passed
        assert processing_time < 0.3


@pytest.mark.asyncio
async def test_min_interval_still_enforced_when_needed(mock_render_haproxy_templates):
    """Test that min_interval is still enforced for rapid successive triggers."""
    import time

    async with managed_debouncer(
        min_interval=MIN_INTERVAL_LONG,  # 0.5s min interval
        max_interval=MAX_INTERVAL_LONG,  # 10s max interval
    ) as debouncer:
        # First render
        await debouncer.trigger()
        await asyncio.sleep(SLEEP_LONG)  # 0.6s wait for first render
        first_count = mock_render_haproxy_templates.call_count
        assert first_count >= 1

        # Trigger again immediately (within min_interval)
        start_time = time.time()
        await debouncer.trigger()
        await asyncio.sleep(
            SLEEP_LONG + SLEEP_SHORT
        )  # Wait for rate limiting + processing

        # Should have rendered but with rate limiting delay
        assert mock_render_haproxy_templates.call_count > first_count
        total_time = time.time() - start_time
        # Should be approximately min_interval (0.5s) plus some processing time
        assert total_time >= MIN_INTERVAL_LONG * 0.8  # Allow 20% timing tolerance


# Extended tests (merged from test_debouncer_extended.py)


def test_warning_for_very_long_max_interval(mock_render_haproxy_templates):
    """Test that a warning is logged for max_interval > 3600."""

    with patch_debouncer_environment() as mock_logger:
        _ = create_debouncer(
            min_interval=60,
            max_interval=7200,  # 2 hours
        )

        # Check that info was called about long max_interval
        mock_logger.info.assert_any_call(
            "Long max_interval (7200s) - templates may become stale during quiet periods"
        )


@pytest.mark.asyncio
async def test_trigger_without_metrics(mock_render_haproxy_templates):
    """Test trigger method when metrics module import fails."""

    debouncer = create_debouncer(min_interval=1, max_interval=5)

    # Mock the import to fail
    with patch.dict("sys.modules", {"haproxy_template_ic.metrics": None}):
        await debouncer.trigger()

        # Should complete without error
        assert debouncer._change_count == 1
        assert debouncer._last_change_time > 0


@pytest.mark.asyncio
async def test_run_metrics_update_import_error(mock_render_haproxy_templates):
    """Test _run method metrics update when import fails."""

    async with managed_debouncer(min_interval=0.1, max_interval=0.2) as debouncer:
        # Patch the metrics import to fail
        with patch.dict("sys.modules", {"haproxy_template_ic.metrics": None}):
            await debouncer.trigger()
            await asyncio.sleep(0.08)

            # Should have rendered despite import error
            assert mock_render_haproxy_templates.called


@pytest.mark.asyncio
async def test_run_metrics_render_import_error(mock_render_haproxy_templates):
    """Test _run method metrics recording when import fails."""

    async with managed_debouncer(min_interval=0.1, max_interval=0.2) as _debouncer:
        # Let it run for a periodic refresh with no metrics
        with patch.dict("sys.modules", {"haproxy_template_ic.metrics": None}):
            await asyncio.sleep(0.12)  # Wait for max_interval

            # Should have rendered despite import error
            assert mock_render_haproxy_templates.called


@pytest.mark.asyncio
async def test_cancelled_error_during_run(mock_render_haproxy_templates):
    """Test handling of CancelledError in _run method."""

    debouncer = create_debouncer(min_interval=0.1, max_interval=1.0)

    # Manually set _stop to False to enter the loop
    debouncer._stop = False

    # Create a custom task that we can cancel
    task = asyncio.create_task(debouncer._run())

    # Give it time to start
    await asyncio.sleep(0.02)

    # Cancel the task
    task.cancel()

    # Wait for it to complete - should not raise
    try:
        await task
    except asyncio.CancelledError:
        pass  # Expected

    # The method should have logged the cancellation


@pytest.mark.asyncio
async def test_unexpected_error_in_run(mock_render_haproxy_templates):
    """Test handling of unexpected errors in _run method."""

    debouncer = create_debouncer(
        min_interval=0.1, max_interval=0.5, error_retry_delay=0.01
    )

    # Patch asyncio.wait_for to raise an unexpected error once
    call_count = 0
    original_wait_for = asyncio.wait_for

    async def mock_wait_for(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Unexpected error")
        return await original_wait_for(*args, **kwargs)

    with patch("asyncio.wait_for", mock_wait_for):
        with patch_debouncer_logger() as mock_logger:
            await debouncer.start()

            try:
                # Give it time to hit the error and recover
                await asyncio.sleep(0.05)

                # Should have logged the error
                error_calls = [
                    call
                    for call in mock_logger.error.call_args_list
                    if "Unexpected error in debouncer task" in str(call)
                ]
                assert len(error_calls) > 0
            finally:
                await debouncer.stop()


@pytest.mark.asyncio
async def test_stop_with_timeout(mock_render_haproxy_templates):
    """Test stop method when task doesn't stop gracefully."""

    # This test doesn't use the render function mocking since it tests timeout behavior
    # Instead we override the internal _run method to simulate a stuck task
    debouncer = create_debouncer(
        min_interval=0.01,  # Very short so render triggers quickly
        max_interval=10,  # Long max_interval so it won't trigger periodic refresh
        stop_timeout=0.01,  # Short timeout for fast test execution
    )

    # Manually override the _run method to ignore stop signals
    async def stubborn_run():
        # This version ignores the stop flag
        while True:
            try:
                await asyncio.sleep(10)  # Never exits
            except asyncio.CancelledError:
                # Re-raise to let stop() handle it
                raise

    debouncer._run = stubborn_run
    await debouncer.start()

    # Give it a moment to start
    await asyncio.sleep(0.005)

    # Now try to stop - it should timeout and cancel
    with patch_debouncer_logger() as mock_logger:
        await debouncer.stop()

        # Should have warned about ungraceful stop
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if "did not stop gracefully" in str(call)
        ]
        assert len(warning_calls) > 0


@pytest.mark.asyncio
async def test_stop_with_cancelled_error_during_await(mock_render_haproxy_templates):
    """Test stop method handling CancelledError when awaiting task."""

    debouncer = create_debouncer(min_interval=0.1, max_interval=1.0)

    # Start the debouncer normally to have a real task
    await debouncer.start()

    # Wait a bit and then cancel the task to simulate a cancelled task
    await asyncio.sleep(0.02)
    debouncer._task.cancel()

    # Stop should handle the CancelledError gracefully
    await debouncer.stop()

    # Task should be done
    assert debouncer._task.done()
