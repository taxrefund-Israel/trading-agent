import asyncio
import sys

# On Python 3.14, patch the deprecated WindowsSelector policy issue
# Streamlit tries to force WindowsSelectorEventLoopPolicy, but we need ProactorEventLoop for proper Windows networking
if sys.platform == "win32":
    # Prevent Streamlit from changing the event loop policy
    _orig_set_policy = asyncio.set_event_loop_policy
    def _patched_set_policy(policy):
        if "Selector" in type(policy).__name__:
            pass  # Don't switch to Selector - keep default (Proactor in Python 3.12+)
        else:
            _orig_set_policy(policy)
    asyncio.set_event_loop_policy = _patched_set_policy

# Run streamlit
from streamlit.web import cli as stcli
import sys
sys.argv = ["streamlit", "run", "test_streamlit.py", "--server.port", "8905", "--server.headless", "true"]
stcli.main()
