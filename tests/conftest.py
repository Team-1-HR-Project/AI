"""Pytest configuration for isolated unit and endpoint tests.

The normal suite must never create ORM tables or seed data in a reachable
Laravel/Railway database. Tests that need a database define their own isolated
SQLite fixture; live database checks are explicit integration tests.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_isolated_test_suite():
    """Marker fixture: no production or shared-database setup is performed."""
    yield

