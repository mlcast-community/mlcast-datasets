"""Sphinx hooks used only by the documentation build."""

from datetime import datetime, timezone


def _set_dynamic_copyright(app, config):
    year = datetime.now(timezone.utc).year
    config.copyright = f"{year}"


def setup(app):
    app.connect("config-inited", _set_dynamic_copyright)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
