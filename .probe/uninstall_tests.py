

# ---------------------------------------------------------------------------
# GAIA home safety: a misconfigured GAIA_HOME must not become a deletion plan
# ---------------------------------------------------------------------------


class TestGaiaHomeSafetyGuard:
    """``_safe_roots`` lists the GAIA home as an allowed root, so the
    containment guard in ``_remove_path`` is vacuous when the home itself is
    wrong. ``GAIA_HOME=$HOME`` turns ``--purge`` into ``$HOME/venv`` +
    ``$HOME/documents`` — and ``documents`` resolves case-insensitively onto
    the real ``Documents`` folder on Windows and APFS.
    """

    def test_gaia_home_at_user_home_refuses_purge_plan(self, fake_home, monkeypatch):
        monkeypatch.setenv("GAIA_HOME", str(fake_home))
        docs = fake_home / "Documents"
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "thesis.docx").write_text("my life's work")

        with pytest.raises(uc.UnsafeGaiaHomeError, match="GAIA_HOME"):
            uc.build_plan(
                venv=False,
                purge=True,
                purge_lemonade=False,
                purge_models=False,
            )

        assert (docs / "thesis.docx").exists()

    def test_gaia_home_at_user_home_refuses_venv_plan(self, fake_home, monkeypatch):
        """Tier 2 deletes ``<home>/venv``, so it needs the same structural guard."""
        monkeypatch.setenv("GAIA_HOME", str(fake_home))

        with pytest.raises(uc.UnsafeGaiaHomeError, match="home directory"):
            uc.build_plan(
                venv=True,
                purge=False,
                purge_lemonade=False,
                purge_models=False,
            )

    def test_gaia_home_above_user_home_is_refused(self, fake_home, monkeypatch):
        monkeypatch.setenv("GAIA_HOME", str(fake_home.parent))

        with pytest.raises(uc.UnsafeGaiaHomeError, match="home directory"):
            uc.build_plan(
                venv=False,
                purge=True,
                purge_lemonade=False,
                purge_models=False,
            )

    def test_gaia_home_at_filesystem_root_is_refused(self, fake_home, monkeypatch):
        root = Path(fake_home.anchor or "/")
        monkeypatch.setenv("GAIA_HOME", str(root))

        with pytest.raises(uc.UnsafeGaiaHomeError, match="root"):
            uc.build_plan(
                venv=False,
                purge=True,
                purge_lemonade=False,
                purge_models=False,
            )

    def test_purge_of_unmarked_directory_is_refused(self, fake_home, monkeypatch):
        """A GAIA_HOME with no sign that GAIA owns it is not purgeable."""
        somewhere = fake_home / "Documents"
        somewhere.mkdir(parents=True, exist_ok=True)
        (somewhere / "thesis.docx").write_text("my life's work")
        monkeypatch.setenv("GAIA_HOME", str(somewhere))

        with pytest.raises(uc.UnsafeGaiaHomeError, match="does not look like"):
            uc.build_plan(
                venv=False,
                purge=True,
                purge_lemonade=False,
                purge_models=False,
            )

        assert (somewhere / "thesis.docx").exists()

    def test_marker_file_makes_an_alternate_home_purgeable(
        self, fake_home, monkeypatch
    ):
        alt = fake_home / "gaia-data"
        alt.mkdir(parents=True, exist_ok=True)
        (alt / "config.json").write_text("{}")
        monkeypatch.setenv("GAIA_HOME", str(alt))

        plan = uc.build_plan(
            venv=False,
            purge=True,
            purge_lemonade=False,
            purge_models=False,
        )
        assert alt / "documents" in plan.unique_paths()

    def test_default_dot_gaia_home_is_always_purgeable(self, fake_home, monkeypatch):
        """The default location is ours by name — no marker file required."""
        monkeypatch.delenv("GAIA_HOME", raising=False)
        (fake_home / ".gaia").mkdir(parents=True, exist_ok=True)

        plan = uc.build_plan(
            venv=False,
            purge=True,
            purge_lemonade=False,
            purge_models=False,
            home=fake_home,
        )
        assert fake_home / ".gaia" / "documents" in plan.unique_paths()

    def test_run_reports_usage_exit_code_and_names_the_root(
        self, fake_home, monkeypatch
    ):
        monkeypatch.setenv("GAIA_HOME", str(fake_home))
        docs = fake_home / "Documents"
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "thesis.docx").write_text("my life's work")

        captured = _Capture()
        exit_code = uc.run(_ns(purge=True, yes=True), printer=captured)

        assert exit_code == uc.EXIT_USAGE, captured.text
        assert "Refusing to uninstall" in captured.text
        assert str(fake_home) in captured.text
        assert (docs / "thesis.docx").exists()

    def test_dry_run_purge_is_refused_too(self, fake_home, monkeypatch):
        """``--dry-run`` must not print a plan we would refuse to execute."""
        monkeypatch.setenv("GAIA_HOME", str(fake_home))

        captured = _Capture()
        exit_code = uc.run(_ns(purge=True, dry_run=True, yes=True), printer=captured)

        assert exit_code == uc.EXIT_USAGE, captured.text
        assert "Would remove" not in captured.text

    def test_plan_output_shows_the_resolved_gaia_home(self, fake_home, monkeypatch):
        monkeypatch.delenv("GAIA_HOME", raising=False)
        _seed_gaia_tree(fake_home)

        captured = _Capture()
        exit_code = uc.run(
            _ns(purge=True, dry_run=True, yes=True), home=fake_home, printer=captured
        )

        assert exit_code == uc.EXIT_OK, captured.text
        assert f"GAIA home: {fake_home / '.gaia'}" in captured.text


# ---------------------------------------------------------------------------
# Symlinked / junctioned entries under ~/.gaia
# ---------------------------------------------------------------------------


class TestSymlinkedEntries:
    """A relocated ``~/.gaia/documents`` used to abort the purge: the
    containment guard resolved the link *target*, which lives outside the
    allowed roots, and the RuntimeError escaped through ``execute_plan``
    after ``venv`` and ``chat`` were already gone.
    """

    def test_symlinked_entry_removes_the_link_not_the_target(self, fake_home, fs):
        _seed_gaia_tree(fake_home)
        gaia = fake_home / ".gaia"
        shutil.rmtree(gaia / "documents")

        target = fake_home / "elsewhere" / "docs"
        target.mkdir(parents=True, exist_ok=True)
        (target / "keep.txt").write_text("keep me")
        fs.create_symlink(gaia / "documents", target)

        ok = uc._remove_path(
            gaia / "documents", allowed_roots=uc._safe_roots(home=fake_home)
        )

        assert ok
        assert not (gaia / "documents").is_symlink()
        assert (target / "keep.txt").exists(), "the link target must survive"

    def test_purge_with_a_symlinked_documents_completes(self, fake_home, fs):
        _seed_gaia_tree(fake_home)
        gaia = fake_home / ".gaia"
        shutil.rmtree(gaia / "documents")
        target = fake_home / "elsewhere" / "docs"
        target.mkdir(parents=True, exist_ok=True)
        (target / "keep.txt").write_text("keep me")
        fs.create_symlink(gaia / "documents", target)

        captured = _Capture()
        exit_code = uc.run(_ns(purge=True, yes=True), home=fake_home, printer=captured)

        assert exit_code == uc.EXIT_OK, captured.text
        assert not (gaia / "venv").exists()
        assert not (gaia / "electron-install.log").exists(), "purge ran to completion"
        assert (target / "keep.txt").exists()

    def test_containment_refusal_does_not_abort_the_rest_of_the_plan(self, fake_home):
        """A refusal is reported and downgrades the exit code — it must not
        raise through ``execute_plan`` and leave a half-deleted tree."""
        _seed_gaia_tree(fake_home)
        gaia = fake_home / ".gaia"
        outside = fake_home / "not-gaia" / "important.txt"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text("do not delete me")

        plan = uc.UninstallPlan()
        plan.tiered_paths.append(("--purge", outside))
        plan.tiered_paths.append(("--purge", gaia / "venv"))

        captured = _Capture()
        exit_code = uc.execute_plan(
            plan, allowed_roots=uc._safe_roots(home=fake_home), printer=captured
        )

        assert exit_code == uc.EXIT_FS_ERROR, captured.text
        assert "outside allowed roots" in captured.text
        assert outside.exists(), "the out-of-root path must be preserved"
        assert not (gaia / "venv").exists(), "later paths must still be processed"
