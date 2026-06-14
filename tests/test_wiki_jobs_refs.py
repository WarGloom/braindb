from braindb.services import wiki_jobs


class FakeCursor:
    def __init__(self):
        self.calls = []
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        if "FROM entities" in sql:
            self.rows = [("11111111-1111-1111-1111-111111111111",)]
        elif "FROM relations" in sql:
            self.rows = []

    def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self):
        self.cursor_obj = FakeCursor()

    def cursor(self):
        return self.cursor_obj


def test_reconcile_summarises_skips_missing_refs():
    conn = FakeConn()
    body = "[[ref:11111111-1111-1111-1111-111111111111]] [[ref:22222222-2222-2222-2222-222222222222]]"

    result = wiki_jobs.reconcile_summarises_additive(
        conn,
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        body,
    )

    inserts = [call for call in conn.cursor_obj.calls if "INSERT INTO relations" in call[0]]

    assert result == {"relations_added": 1, "relations_removed": 0}
    assert len(inserts) == 1
    assert inserts[0][1][1] == "11111111-1111-1111-1111-111111111111"
