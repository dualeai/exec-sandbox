"""Unit tests for the exec-sandbox CLI.

Tests CLI argument parsing, help output, and error handling.
Uses Click's CliRunner for isolated testing without running actual VMs.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from exec_sandbox import ExecutionResult, Language, TimingBreakdown, __version__
from exec_sandbox.assets import PrefetchResult
from exec_sandbox.cli import (
    EXIT_CLI_ERROR,
    EXIT_SANDBOX_ERROR,
    EXIT_SUCCESS,
    EXIT_TIMEOUT,
    MultiSourceResult,
    SourceInput,
    _compute_multi_exit_code,
    cli,
    detect_language,
    format_error,
    format_multi_result_json,
    format_result_json,
    parse_download_spec,
    parse_env_vars,
    parse_upload_spec,
    prefetch_command,
    run_command,
    truncate_source,
)


# ============================================================================
# Test Fixtures
# ============================================================================
@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_result() -> ExecutionResult:
    """Create a mock execution result for testing."""
    return ExecutionResult(
        stdout="hello\n",
        stderr="",
        exit_code=0,
        execution_time_ms=42,
        timing=TimingBreakdown(
            setup_ms=10,
            boot_ms=300,
            execute_ms=70,
            total_ms=420,
        ),
        warm_pool_hit=False,
    )


# ============================================================================
# Unit Tests - Helper Functions
# ============================================================================
class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_python_extension(self) -> None:
        """Detects Python from .py extension."""
        assert detect_language("script.py") == "python"
        assert detect_language("/path/to/script.py") == "python"
        assert detect_language("test.PY") == "python"  # Case insensitive

    def test_javascript_extensions(self) -> None:
        """Detects JavaScript from .js and .mjs extensions."""
        assert detect_language("app.js") == "javascript"
        assert detect_language("module.mjs") == "javascript"

    def test_shell_extension(self) -> None:
        """Detects raw/shell from .sh extension."""
        assert detect_language("run.sh") == "raw"

    def test_unknown_extension(self) -> None:
        """Returns None for unknown extensions."""
        assert detect_language("file.txt") is None
        assert detect_language("file.go") is None
        assert detect_language("file.rs") is None

    def test_no_extension(self) -> None:
        """Returns None for files without extension."""
        assert detect_language("Makefile") is None
        assert detect_language("README") is None

    def test_stdin_marker(self) -> None:
        """Returns None for stdin marker."""
        assert detect_language("-") is None

    def test_none_input(self) -> None:
        """Returns None for None input."""
        assert detect_language(None) is None


class TestParseEnvVars:
    """Tests for parse_env_vars function."""

    def test_single_var(self) -> None:
        """Parses single environment variable."""
        result = parse_env_vars(("KEY=value",))
        assert result == {"KEY": "value"}

    def test_multiple_vars(self) -> None:
        """Parses multiple environment variables."""
        result = parse_env_vars(("KEY1=value1", "KEY2=value2"))
        assert result == {"KEY1": "value1", "KEY2": "value2"}

    def test_value_with_equals(self) -> None:
        """Handles values containing equals sign."""
        result = parse_env_vars(("KEY=a=b=c",))
        assert result == {"KEY": "a=b=c"}

    def test_empty_value(self) -> None:
        """Handles empty values."""
        result = parse_env_vars(("KEY=",))
        assert result == {"KEY": ""}

    def test_empty_tuple(self) -> None:
        """Handles empty input."""
        result = parse_env_vars(())
        assert result == {}

    def test_invalid_format_no_equals(self) -> None:
        """Raises error for invalid format."""
        import click

        with pytest.raises(click.BadParameter, match="Invalid format"):
            parse_env_vars(("INVALID",))

    def test_invalid_format_empty_key(self) -> None:
        """Raises error for empty key."""
        import click

        with pytest.raises(click.BadParameter, match="Empty key"):
            parse_env_vars(("=value",))


class TestFormatError:
    """Tests for format_error function."""

    def test_basic_error(self) -> None:
        """Formats basic error message."""
        result = format_error("Test error", "This is the explanation")
        assert "Error: Test error" in result
        assert "This is the explanation" in result

    def test_error_with_suggestions(self) -> None:
        """Formats error with suggestions."""
        result = format_error(
            "Test error",
            "This is why",
            ["Try this", "Or try that"],
        )
        assert "Suggestions:" in result
        assert "Try this" in result
        assert "Or try that" in result


class TestFormatResultJson:
    """Tests for format_result_json function."""

    def test_json_format(self, mock_result: ExecutionResult) -> None:
        """Formats result as valid JSON."""
        import json

        result = format_result_json(mock_result)
        parsed = json.loads(result)

        assert parsed["stdout"] == "hello\n"
        assert parsed["stderr"] == ""
        assert parsed["exit_code"] == 0
        assert parsed["timing"]["total_ms"] == 420
        assert parsed["warm_pool_hit"] is False


# ============================================================================
# CLI Command Tests
# ============================================================================
class TestCliHelp:
    """Tests for CLI help output."""

    def test_help_flag(self, runner: CliRunner) -> None:
        """Shows help with --help flag."""
        result = runner.invoke(run_command, ["--help"])
        assert result.exit_code == 0
        assert "Execute code in an isolated VM sandbox" in result.output
        assert "--language" in result.output
        assert "--timeout" in result.output

    def test_short_help_flag(self, runner: CliRunner) -> None:
        """Shows help with -h flag."""
        result = runner.invoke(run_command, ["-h"])
        assert result.exit_code == 0
        assert "Execute code" in result.output


class TestCliVersion:
    """Tests for CLI version output."""

    def test_version_flag(self, runner: CliRunner) -> None:
        """Shows version with --version flag."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output


class TestCliArgumentParsing:
    """Tests for CLI argument parsing (without running actual code)."""

    def test_no_args_shows_error(self, runner: CliRunner) -> None:
        """Shows error when no arguments provided."""
        result = runner.invoke(run_command, [])
        assert result.exit_code == EXIT_CLI_ERROR
        assert "No code provided" in result.output

    def test_empty_code_shows_error(self, runner: CliRunner) -> None:
        """Shows error for empty code."""
        # Empty string argument passes through but results in empty code
        result = runner.invoke(run_command, [""])
        assert result.exit_code == EXIT_CLI_ERROR
        # Either "Empty code" or "No code" depending on how Click handles empty strings
        assert "Empty code" in result.output or "No code" in result.output

    def test_whitespace_only_code_shows_error(self, runner: CliRunner) -> None:
        """Shows error for whitespace-only code."""
        result = runner.invoke(run_command, ["   "])
        assert result.exit_code == EXIT_CLI_ERROR
        assert "Empty code provided" in result.output


class TestCliCodeExecution:
    """Tests for CLI code execution with mocked scheduler."""

    def test_inline_code(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Executes inline code."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["print('hello')"])

            assert result.exit_code == 0
            mock_scheduler.run.assert_called_once()
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["code"] == "print('hello')"
            assert call_kwargs["language"] == Language.PYTHON

    def test_explicit_language(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Uses explicitly specified language."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["-l", "javascript", "console.log('hi')"])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["language"] == Language.JAVASCRIPT

    def test_code_flag(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Executes code from -c flag."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["-c", "print(1)"])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["code"] == "print(1)"

    def test_packages(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Passes packages to scheduler."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(
                run_command,
                ["-p", "requests", "-p", "pandas==2.2.0", "import requests"],
            )

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["packages"] == ["requests", "pandas==2.2.0"]

    def test_env_vars(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Passes environment variables to scheduler."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(
                run_command,
                ["-e", "KEY=value", "-e", "DEBUG=1", "print(1)"],
            )

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["env_vars"] == {"KEY": "value", "DEBUG": "1"}

    def test_timeout_and_memory(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Passes timeout and memory settings."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["-t", "60", "-m", "512", "print(1)"])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["timeout_seconds"] == 60
            assert call_kwargs["memory_mb"] == 512

    def test_network_options(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Passes network options to scheduler."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(
                run_command,
                [
                    "--network",
                    "--allow-domain",
                    "api.example.com",
                    "--allow-domain",
                    "cdn.example.com",
                    "print(1)",
                ],
            )

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["allow_network"] is True
            assert call_kwargs["allowed_domains"] == ["api.example.com", "cdn.example.com"]

    def test_json_output(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Outputs JSON when --json flag is used."""
        import json

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["--json", "print(1)"])

            assert result.exit_code == 0
            # Should be valid JSON
            parsed = json.loads(result.output)
            assert "stdout" in parsed
            assert "exit_code" in parsed
            assert "timing" in parsed


class TestCliFileInput:
    """Tests for CLI file input."""

    def test_file_input(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """Reads code from file."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("print('from file')")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, [str(test_file)])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["code"] == "print('from file')"
            # Language should be auto-detected from .py extension
            assert call_kwargs["language"] == Language.PYTHON

    def test_file_language_detection(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """Auto-detects language from file extension."""
        # Create a JS file
        test_file = tmp_path / "app.js"
        test_file.write_text("console.log('hello')")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, [str(test_file)])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["language"] == Language.JAVASCRIPT


class TestCliStdinInput:
    """Tests for CLI stdin input."""

    def test_stdin_input(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Reads code from stdin."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["-"], input="print('from stdin')")

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["code"] == "print('from stdin')"


class TestCliErrorHandling:
    """Tests for CLI error handling."""

    def test_package_not_allowed_error(self, runner: CliRunner) -> None:
        """Handles PackageNotAllowedError."""
        from exec_sandbox import PackageNotAllowedError

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.side_effect = PackageNotAllowedError("fake-package")
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["import fake"])

            assert result.exit_code == EXIT_CLI_ERROR
            assert "not allowed" in result.output.lower()

    def test_timeout_error(self, runner: CliRunner) -> None:
        """Handles VmTimeoutError."""
        from exec_sandbox import VmTimeoutError

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.side_effect = VmTimeoutError("Execution timed out")
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["import time; time.sleep(100)"])

            assert result.exit_code == EXIT_TIMEOUT
            assert "timed out" in result.output.lower()

    def test_sandbox_error(self, runner: CliRunner) -> None:
        """Handles SandboxError."""
        from exec_sandbox import SandboxError

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.side_effect = SandboxError("VM boot failed")
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["print(1)"])

            assert result.exit_code == EXIT_SANDBOX_ERROR
            assert "sandbox error" in result.output.lower()


class TestCliNoValidation:
    """Tests for --no-validation flag."""

    def test_no_validation_flag(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Disables package validation with --no-validation."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["--no-validation", "print(1)"])

            assert result.exit_code == 0
            # Check that config was created with enable_package_validation=False
            config = scheduler_cls.call_args[0][0]
            assert config.enable_package_validation is False


# ============================================================================
# Edge Cases and Boundary Tests
# ============================================================================
class TestCliVersionShortFlag:
    """Tests for -V version flag."""

    def test_short_version_flag(self, runner: CliRunner) -> None:
        """Shows version with -V flag."""
        result = runner.invoke(cli, ["-V"])
        assert result.exit_code == 0
        assert __version__ in result.output


class TestCliLanguageEdgeCases:
    """Tests for language option edge cases."""

    def test_language_case_insensitive(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Language option is case insensitive."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["-l", "PYTHON", "print(1)"])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["language"] == Language.PYTHON

    def test_invalid_language(self, runner: CliRunner) -> None:
        """Rejects invalid language."""
        result = runner.invoke(run_command, ["-l", "ruby", "puts 'hello'"])
        assert result.exit_code == EXIT_CLI_ERROR
        assert "ruby" in result.output.lower() or "invalid" in result.output.lower()

    def test_raw_language(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Executes with raw language."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["-l", "raw", "echo hello"])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["language"] == Language.RAW


class TestCliFileEdgeCases:
    """Tests for file input edge cases."""

    def test_nonexistent_file_treated_as_code(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Non-existent file path is treated as inline code."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            # This looks like a file but doesn't exist, so treated as code
            result = runner.invoke(run_command, ["/nonexistent/script.py"])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            # The path string is treated as code
            assert call_kwargs["code"] == "/nonexistent/script.py"

    def test_directory_treated_as_code(self, runner: CliRunner, mock_result: ExecutionResult, tmp_path: Path) -> None:
        """Directory path is treated as inline code (not a file)."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, [str(tmp_path)])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            # Directory path is treated as code string
            assert call_kwargs["code"] == str(tmp_path)

    def test_file_with_multiple_extensions(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """File with multiple extensions uses last extension."""
        test_file = tmp_path / "test.backup.py"
        test_file.write_text("print('multi ext')")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, [str(test_file)])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["language"] == Language.PYTHON

    def test_hidden_file(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """Hidden file is read correctly."""
        test_file = tmp_path / ".hidden.py"
        test_file.write_text("print('hidden')")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, [str(test_file)])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["code"] == "print('hidden')"

    def test_file_with_spaces_in_path(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """File with spaces in path is handled correctly."""
        subdir = tmp_path / "my scripts"
        subdir.mkdir()
        test_file = subdir / "my script.py"
        test_file.write_text("print('spaces')")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, [str(test_file)])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["code"] == "print('spaces')"


class TestCliCodeEdgeCases:
    """Tests for code content edge cases."""

    def test_code_with_quotes(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Code containing quotes is handled correctly."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["print(\"hello 'world'\")"])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["code"] == "print(\"hello 'world'\")"

    def test_code_with_newlines(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Code containing newlines is handled correctly."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            code = "x = 1\nprint(x)"
            result = runner.invoke(run_command, [code])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["code"] == code

    def test_code_with_unicode(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Code containing unicode is handled correctly."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            code = "print('Hello 世界 🌍')"
            result = runner.invoke(run_command, [code])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["code"] == code

    def test_code_flag_and_positional_both_run(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Both -c flag and positional arguments run as separate sources."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            # Both -c and positional provided - both should run
            result = runner.invoke(run_command, ["-c", "print('from -c')", "print('from pos')"])

            assert result.exit_code == 0
            # Both sources should be executed (multi-input mode)
            assert mock_scheduler.run.call_count == 2
            # -c sources come first, then positional
            calls = mock_scheduler.run.call_args_list
            codes = [call.kwargs["code"] for call in calls]
            assert "print('from -c')" in codes
            assert "print('from pos')" in codes


class TestCliStdinEdgeCases:
    """Tests for stdin input edge cases."""

    def test_stdin_empty_input(self, runner: CliRunner) -> None:
        """Empty stdin shows error."""
        result = runner.invoke(run_command, ["-"], input="")
        assert result.exit_code == EXIT_CLI_ERROR
        assert "Empty code provided" in result.output

    def test_stdin_whitespace_only(self, runner: CliRunner) -> None:
        """Whitespace-only stdin shows error."""
        result = runner.invoke(run_command, ["-"], input="   \n\t\n   ")
        assert result.exit_code == EXIT_CLI_ERROR
        assert "Empty code provided" in result.output

    def test_stdin_with_explicit_language(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Stdin with explicit language works."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["-l", "javascript", "-"], input="console.log('stdin')")

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["language"] == Language.JAVASCRIPT


class TestCliEnvVarEdgeCases:
    """Tests for environment variable edge cases."""

    def test_env_var_special_characters_in_value(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Env var with special characters in value."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["-e", "KEY=val!@#$%^&*()", "print(1)"])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["env_vars"]["KEY"] == "val!@#$%^&*()"

    def test_env_var_with_quotes_in_value(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Env var with quotes in value."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["-e", 'KEY="quoted value"', "print(1)"])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["env_vars"]["KEY"] == '"quoted value"'

    def test_env_var_numeric_key(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Env var with numeric key."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["-e", "123=value", "print(1)"])

            assert result.exit_code == 0
            call_kwargs = mock_scheduler.run.call_args.kwargs
            assert call_kwargs["env_vars"]["123"] == "value"


class TestCliTimeoutMemoryEdgeCases:
    """Tests for timeout and memory edge cases."""

    def test_invalid_timeout_string(self, runner: CliRunner) -> None:
        """Invalid timeout string shows error."""
        result = runner.invoke(run_command, ["-t", "abc", "print(1)"])
        assert result.exit_code == EXIT_CLI_ERROR

    def test_negative_timeout(self, runner: CliRunner) -> None:
        """Negative timeout is rejected by Click (not a valid integer option)."""
        result = runner.invoke(run_command, ["-t", "-1", "print(1)"])
        # Click treats -1 as another flag, not a value, causing an error
        assert result.exit_code != 0

    def test_zero_timeout(self, runner: CliRunner) -> None:
        """Zero timeout fails SchedulerConfig validation."""
        result = runner.invoke(run_command, ["-t", "0", "print(1)"])
        # SchedulerConfig requires timeout >= 1, Pydantic raises ValidationError
        assert result.exit_code != 0

    def test_invalid_memory_string(self, runner: CliRunner) -> None:
        """Invalid memory string shows error."""
        result = runner.invoke(run_command, ["-m", "abc", "print(1)"])
        assert result.exit_code == EXIT_CLI_ERROR


class TestCliExitCodePassthrough:
    """Tests for exit code passthrough from executed code."""

    def test_nonzero_exit_code(self, runner: CliRunner) -> None:
        """Non-zero exit code is passed through."""
        result_with_error = ExecutionResult(
            stdout="",
            stderr="error",
            exit_code=42,
            execution_time_ms=10,
            timing=TimingBreakdown(setup_ms=1, boot_ms=10, execute_ms=5, total_ms=16),
            warm_pool_hit=False,
        )

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = result_with_error
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["exit(42)"])

            assert result.exit_code == 42


class TestCliJsonOutputEdgeCases:
    """Tests for JSON output edge cases."""

    def test_json_output_with_memory_peak(self, runner: CliRunner) -> None:
        """JSON output includes memory peak when present."""
        import json

        result_with_memory = ExecutionResult(
            stdout="hello\n",
            stderr="",
            exit_code=0,
            execution_time_ms=42,
            external_memory_peak_mb=128,
            timing=TimingBreakdown(setup_ms=10, boot_ms=300, execute_ms=70, total_ms=420),
            warm_pool_hit=False,
        )

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = result_with_memory
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["--json", "print(1)"])

            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert parsed["memory_peak_mb"] == 128

    def test_json_output_with_nonzero_exit(self, runner: CliRunner) -> None:
        """JSON output works with non-zero exit code."""
        import json

        result_with_error = ExecutionResult(
            stdout="",
            stderr="error message",
            exit_code=1,
            execution_time_ms=10,
            timing=TimingBreakdown(setup_ms=1, boot_ms=10, execute_ms=5, total_ms=16),
            warm_pool_hit=False,
        )

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = result_with_error
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["--json", "exit(1)"])

            assert result.exit_code == 1
            parsed = json.loads(result.output)
            assert parsed["exit_code"] == 1
            assert parsed["stderr"] == "error message"


class TestCliQuietMode:
    """Tests for quiet mode."""

    def test_quiet_mode(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Quiet mode suppresses progress output."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["-q", "print(1)"])

            assert result.exit_code == 0
            # In quiet mode, no "Done in Xms" footer
            assert "Done in" not in result.output


class TestDetectLanguageEdgeCases:
    """Additional edge cases for detect_language."""

    def test_uppercase_extension(self) -> None:
        """Handles uppercase extensions."""
        assert detect_language("SCRIPT.PY") == "python"
        assert detect_language("APP.JS") == "javascript"

    def test_mixed_case_extension(self) -> None:
        """Handles mixed case extensions."""
        assert detect_language("script.Py") == "python"
        assert detect_language("app.Js") == "javascript"

    def test_double_extension(self) -> None:
        """Uses last extension for double extensions."""
        assert detect_language("file.tar.py") == "python"
        assert detect_language("file.py.bak") is None  # .bak is unknown

    def test_dotfiles_have_no_extension(self) -> None:
        """Dotfiles (like .py, .js) are hidden files, not extension-only files."""
        # .py is a hidden file named ".py", not a file with extension .py
        assert detect_language(".py") is None
        assert detect_language(".js") is None
        # But .hidden.py IS a hidden file with .py extension
        assert detect_language(".hidden.py") == "python"

    def test_path_with_dots_in_directory(self) -> None:
        """Handles paths with dots in directory names."""
        assert detect_language("/path.to.dir/script.py") == "python"
        assert detect_language("./my.project/app.js") == "javascript"


class TestParseEnvVarsEdgeCases:
    """Additional edge cases for parse_env_vars."""

    def test_unicode_in_key_and_value(self) -> None:
        """Handles unicode in keys and values."""
        result = parse_env_vars(("キー=値",))
        assert result == {"キー": "値"}

    def test_very_long_value(self) -> None:
        """Handles very long values."""
        long_value = "x" * 10000
        result = parse_env_vars((f"KEY={long_value}",))
        assert result == {"KEY": long_value}

    def test_value_with_newlines(self) -> None:
        """Handles values with newlines."""
        result = parse_env_vars(("KEY=line1\nline2",))
        assert result == {"KEY": "line1\nline2"}

    def test_duplicate_keys(self) -> None:
        """Later duplicate key overwrites earlier."""
        result = parse_env_vars(("KEY=first", "KEY=second"))
        assert result == {"KEY": "second"}


# ============================================================================
# Multi-Input Tests (V2)
# ============================================================================
class TestParseUploadSpec:
    """Tests for parse_upload_spec function."""

    # --- Normal cases ---

    def test_basic_format(self, tmp_path: Path) -> None:
        """Parses LOCAL:GUEST format."""
        local_file = tmp_path / "data.csv"
        local_file.write_text("content")
        local_path, guest_path = parse_upload_spec(f"{local_file}:data.csv")
        assert local_path == str(local_file)
        assert guest_path == "data.csv"

    def test_guest_with_subdirectory(self, tmp_path: Path) -> None:
        """Parses guest path with subdirectory."""
        local_file = tmp_path / "input.txt"
        local_file.write_text("content")
        local_path, guest_path = parse_upload_spec(f"{local_file}:subdir/input.txt")
        assert local_path == str(local_file)
        assert guest_path == "subdir/input.txt"

    def test_relative_local_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Parses relative local path."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rel.csv").write_text("content")
        local_path, guest_path = parse_upload_spec("rel.csv:data.csv")
        assert local_path == "rel.csv"
        assert guest_path == "data.csv"

    def test_absolute_local_path(self, tmp_path: Path) -> None:
        """Parses absolute local path."""
        local_file = tmp_path / "abs.txt"
        local_file.write_text("content")
        local_path, guest_path = parse_upload_spec(f"{local_file}:dest.txt")
        assert local_path == str(local_file)
        assert guest_path == "dest.txt"

    def test_symlink_local_path(self, tmp_path: Path) -> None:
        """Follows symlink for local path (exists check passes)."""
        real = tmp_path / "real.txt"
        real.write_text("content")
        link = tmp_path / "link.txt"
        link.symlink_to(real)
        local_path, guest_path = parse_upload_spec(f"{link}:data.txt")
        assert local_path == str(link)
        assert guest_path == "data.txt"

    # --- Edge cases ---

    def test_multiple_colons_splits_on_first(self, tmp_path: Path) -> None:
        """Splits on first colon — guest path may contain colons."""
        local_file = tmp_path / "f.txt"
        local_file.write_text("content")
        local_path, guest_path = parse_upload_spec(f"{local_file}:a:b:c")
        assert local_path == str(local_file)
        assert guest_path == "a:b:c"

    def test_spaces_in_local_path(self, tmp_path: Path) -> None:
        """Handles spaces in local path."""
        local_file = tmp_path / "my file.csv"
        local_file.write_text("content")
        local_path, guest_path = parse_upload_spec(f"{local_file}:data.csv")
        assert local_path == str(local_file)
        assert guest_path == "data.csv"

    def test_spaces_in_guest_path(self, tmp_path: Path) -> None:
        """Handles spaces in guest path."""
        local_file = tmp_path / "f.txt"
        local_file.write_text("content")
        _, guest_path = parse_upload_spec(f"{local_file}:my data.csv")
        assert guest_path == "my data.csv"

    def test_unicode_in_paths(self, tmp_path: Path) -> None:
        """Handles unicode characters in paths."""
        local_file = tmp_path / "données.csv"
        local_file.write_text("content")
        local_path, guest_path = parse_upload_spec(f"{local_file}:数据.csv")
        assert local_path == str(local_file)
        assert guest_path == "数据.csv"

    def test_dotfile_local(self, tmp_path: Path) -> None:
        """Handles dotfile as local path."""
        local_file = tmp_path / ".hidden"
        local_file.write_text("secret")
        local_path, _ = parse_upload_spec(f"{local_file}:config")
        assert local_path == str(local_file)

    # --- Error cases ---

    def test_missing_colon(self) -> None:
        """Rejects spec without colon."""
        import click

        with pytest.raises(click.BadParameter, match="Invalid upload format"):
            parse_upload_spec("data.csv")

    def test_empty_guest_path(self, tmp_path: Path) -> None:
        """Rejects empty guest path (trailing colon)."""
        import click

        local_file = tmp_path / "data.csv"
        local_file.write_text("content")
        with pytest.raises(click.BadParameter, match="Invalid upload format"):
            parse_upload_spec(f"{local_file}:")

    def test_empty_local_path(self) -> None:
        """Rejects empty local path (leading colon)."""
        import click

        with pytest.raises(click.BadParameter, match="Empty local path"):
            parse_upload_spec(":data.csv")

    def test_nonexistent_local_file(self) -> None:
        """Rejects non-existent local file."""
        import click

        with pytest.raises(click.BadParameter, match="Local file not found"):
            parse_upload_spec("/nonexistent/file.csv:data.csv")

    def test_just_colon(self) -> None:
        """Rejects bare colon."""
        import click

        with pytest.raises(click.BadParameter):
            parse_upload_spec(":")

    def test_double_colon(self) -> None:
        """Rejects double colon (empty local)."""
        import click

        with pytest.raises(click.BadParameter, match="Empty local path"):
            parse_upload_spec("::")

    def test_local_is_directory(self, tmp_path: Path) -> None:
        """Accepts directory as local path (exists returns True for dirs).

        The session.write_file will fail later when trying to read content,
        but parse_upload_spec only checks existence.
        """
        subdir = tmp_path / "mydir"
        subdir.mkdir()
        # Does not raise — existence check passes for directories
        local_path, _ = parse_upload_spec(f"{subdir}:data")
        assert local_path == str(subdir)

    def test_broken_symlink(self, tmp_path: Path) -> None:
        """Rejects broken symlink (target doesn't exist)."""
        import click

        link = tmp_path / "broken"
        link.symlink_to(tmp_path / "nonexistent_target")
        with pytest.raises(click.BadParameter, match="Local file not found"):
            parse_upload_spec(f"{link}:data.txt")


class TestParseDownloadSpec:
    """Tests for parse_download_spec function."""

    # --- Normal cases ---

    def test_full_format(self) -> None:
        """Parses GUEST:LOCAL format."""
        guest_path, local_path = parse_download_spec("output.csv:./results/out.csv")
        assert guest_path == "output.csv"
        assert local_path == "./results/out.csv"

    def test_shorthand_guest_only(self) -> None:
        """Parses GUEST shorthand (no colon) -> local = basename."""
        guest_path, local_path = parse_download_spec("output.csv")
        assert guest_path == "output.csv"
        assert local_path == "output.csv"

    def test_shorthand_guest_with_subdir(self) -> None:
        """Parses GUEST with subdir -> local = basename only."""
        guest_path, local_path = parse_download_spec("subdir/output.csv")
        assert guest_path == "subdir/output.csv"
        assert local_path == "output.csv"

    def test_local_absolute_path(self) -> None:
        """Parses absolute local path."""
        guest_path, local_path = parse_download_spec("data.csv:/tmp/out.csv")
        assert guest_path == "data.csv"
        assert local_path == "/tmp/out.csv"

    def test_local_with_nested_dirs(self) -> None:
        """Parses local path with nested directories."""
        guest_path, local_path = parse_download_spec("result.bin:./deep/nested/dir/out.bin")
        assert guest_path == "result.bin"
        assert local_path == "./deep/nested/dir/out.bin"

    # --- Edge cases ---

    def test_multiple_colons_splits_on_first(self) -> None:
        """Splits on first colon — local path may contain colons."""
        guest_path, local_path = parse_download_spec("data.csv:a:b:c")
        assert guest_path == "data.csv"
        assert local_path == "a:b:c"

    def test_spaces_in_guest_path(self) -> None:
        """Handles spaces in guest path."""
        guest_path, local_path = parse_download_spec("my file.csv:out.csv")
        assert guest_path == "my file.csv"
        assert local_path == "out.csv"

    def test_spaces_in_local_path(self) -> None:
        """Handles spaces in local path."""
        guest_path, local_path = parse_download_spec("data.csv:./my output.csv")
        assert guest_path == "data.csv"
        assert local_path == "./my output.csv"

    def test_unicode_in_paths(self) -> None:
        """Handles unicode characters in paths."""
        guest_path, local_path = parse_download_spec("数据.csv:./données.csv")
        assert guest_path == "数据.csv"
        assert local_path == "./données.csv"

    def test_shorthand_deeply_nested_guest(self) -> None:
        """Shorthand with deep nesting extracts basename."""
        guest_path, local_path = parse_download_spec("a/b/c/d/result.txt")
        assert guest_path == "a/b/c/d/result.txt"
        assert local_path == "result.txt"

    def test_shorthand_dotfile(self) -> None:
        """Shorthand preserves dotfile basename."""
        guest_path, local_path = parse_download_spec(".hidden")
        assert guest_path == ".hidden"
        assert local_path == ".hidden"

    def test_guest_with_trailing_slash_shorthand(self) -> None:
        """Guest path with trailing slash — Path strips it, basename is 'subdir'."""
        guest_path, local_path = parse_download_spec("subdir/")
        assert guest_path == "subdir/"
        assert local_path == "subdir"

    # --- Boundary cases ---

    def test_root_path_shorthand(self) -> None:
        """Guest path '/' — basename is empty, rejected."""
        import click

        with pytest.raises(click.BadParameter, match="Empty local path"):
            parse_download_spec("/")

    def test_dot_shorthand(self) -> None:
        """Guest path '.' — Path('.').name is empty string, caught by validation."""
        import click

        with pytest.raises(click.BadParameter, match="Empty local path"):
            parse_download_spec(".")

    def test_dotdot_shorthand(self) -> None:
        """Guest path '..' — basename is '..', returned as-is (guest agent validates)."""
        guest_path, local_path = parse_download_spec("..")
        assert guest_path == ".."
        assert local_path == ".."

    # --- Error cases ---

    def test_empty_spec(self) -> None:
        """Rejects empty spec."""
        import click

        with pytest.raises(click.BadParameter, match="Empty download path"):
            parse_download_spec("")

    def test_empty_guest_path(self) -> None:
        """Rejects empty guest path (leading colon)."""
        import click

        with pytest.raises(click.BadParameter, match="Empty guest path"):
            parse_download_spec(":./out.csv")

    def test_empty_local_path(self) -> None:
        """Rejects empty local path (trailing colon)."""
        import click

        with pytest.raises(click.BadParameter, match="Empty local path"):
            parse_download_spec("output.csv:")

    def test_just_colon(self) -> None:
        """Rejects bare colon (both sides empty)."""
        import click

        with pytest.raises(click.BadParameter, match="Empty guest path"):
            parse_download_spec(":")

    def test_double_colon(self) -> None:
        """Rejects double colon (empty guest)."""
        import click

        with pytest.raises(click.BadParameter, match="Empty guest path"):
            parse_download_spec("::")


class TestCliFileIoIntegration:
    """Tests for --upload/--download argument flow through CLI to run_code."""

    def test_upload_parsed_and_passed(self, runner: CliRunner, mock_result: ExecutionResult, tmp_path: Path) -> None:
        """Upload spec is parsed and tuple flows to session.write_file."""
        local_file = tmp_path / "data.csv"
        local_file.write_text("content")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_session = AsyncMock()
            mock_session.exec.return_value = mock_result
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None

            mock_scheduler = AsyncMock()
            mock_scheduler.session.return_value = mock_session
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(
                run_command,
                ["--upload", f"{local_file}:input.csv", "-c", "print(1)"],
            )

            assert result.exit_code == 0
            mock_session.write_file.assert_called_once_with("input.csv", local_file)

    def test_download_full_parsed_and_passed(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Download GUEST:LOCAL spec flows correctly to session.read_file."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_session = AsyncMock()
            mock_session.exec.return_value = mock_result

            async def _fake_read(path: str, *, destination: Path) -> None:
                destination.write_bytes(b"downloaded")

            mock_session.read_file.side_effect = _fake_read
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None

            mock_scheduler = AsyncMock()
            mock_scheduler.session.return_value = mock_session
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            with runner.isolated_filesystem():
                result = runner.invoke(
                    run_command,
                    ["--download", "out.csv:./result.csv", "-c", "print(1)"],
                )

                assert result.exit_code == 0
                mock_session.read_file.assert_called_once_with("out.csv", destination=Path("result.csv"))
                assert Path("result.csv").read_bytes() == b"downloaded"

    def test_download_shorthand_writes_to_basename(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Download GUEST shorthand writes to ./basename in cwd."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_session = AsyncMock()
            mock_session.exec.return_value = mock_result

            async def _fake_read(path: str, *, destination: Path) -> None:
                destination.write_bytes(b"data")

            mock_session.read_file.side_effect = _fake_read
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None

            mock_scheduler = AsyncMock()
            mock_scheduler.session.return_value = mock_session
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            with runner.isolated_filesystem():
                result = runner.invoke(
                    run_command,
                    ["--download", "subdir/output.csv", "-c", "print(1)"],
                )

                assert result.exit_code == 0
                mock_session.read_file.assert_called_once_with("subdir/output.csv", destination=Path("output.csv"))
                assert Path("output.csv").read_bytes() == b"data"

    def test_upload_and_download_together(
        self, runner: CliRunner, mock_result: ExecutionResult, tmp_path: Path
    ) -> None:
        """Upload + download in same command uses session path."""
        local_file = tmp_path / "input.txt"
        local_file.write_text("hello")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_session = AsyncMock()
            mock_session.exec.return_value = mock_result

            async def _fake_read(path: str, *, destination: Path) -> None:
                destination.write_bytes(b"result")

            mock_session.read_file.side_effect = _fake_read
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None

            mock_scheduler = AsyncMock()
            mock_scheduler.session.return_value = mock_session
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            with runner.isolated_filesystem():
                result = runner.invoke(
                    run_command,
                    [
                        "--upload",
                        f"{local_file}:input.txt",
                        "--download",
                        "output.txt:result.txt",
                        "-c",
                        "print(1)",
                    ],
                )

                assert result.exit_code == 0
                mock_session.write_file.assert_called_once()
                mock_session.read_file.assert_called_once_with("output.txt", destination=Path("result.txt"))
                assert Path("result.txt").read_bytes() == b"result"

    def test_no_file_io_uses_direct_path(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Without --upload/--download, uses scheduler.run() not session."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["-c", "print(1)"])

            assert result.exit_code == 0
            mock_scheduler.run.assert_called_once()
            mock_scheduler.session.assert_not_called()

    def test_bad_upload_spec_shows_usage_error(self, runner: CliRunner) -> None:
        """Invalid upload spec surfaces as CLI usage error."""
        result = runner.invoke(run_command, ["--upload", "nocolon", "-c", "print(1)"])
        assert result.exit_code == EXIT_CLI_ERROR
        assert "Invalid upload format" in result.output

    def test_bad_download_spec_shows_usage_error(self, runner: CliRunner) -> None:
        """Invalid download spec surfaces as CLI usage error."""
        result = runner.invoke(run_command, ["--download", ":", "-c", "print(1)"])
        assert result.exit_code == EXIT_CLI_ERROR
        assert "Empty guest path" in result.output


class TestTruncateSource:
    """Tests for truncate_source helper."""

    def test_short_source(self) -> None:
        """Short sources are not truncated."""
        assert truncate_source("print(1)") == "print(1)"
        assert truncate_source("short") == "short"

    def test_long_source(self) -> None:
        """Long sources are truncated with ellipsis."""
        long_src = "x" * 50
        result = truncate_source(long_src, max_len=40)
        assert len(result) == 40
        assert result.endswith("...")

    def test_exact_length(self) -> None:
        """Exact length sources are not truncated."""
        src = "x" * 40
        assert truncate_source(src, max_len=40) == src


class TestSourceInput:
    """Tests for SourceInput dataclass."""

    def test_create_source_input(self) -> None:
        """Creates SourceInput with required fields."""
        src = SourceInput(code="print(1)", label="test", language=Language.PYTHON)
        assert src.code == "print(1)"
        assert src.label == "test"
        assert src.language == Language.PYTHON


class TestMultiSourceResult:
    """Tests for MultiSourceResult dataclass."""

    def test_create_success_result(self, mock_result: ExecutionResult) -> None:
        """Creates successful MultiSourceResult."""
        result = MultiSourceResult(
            index=0,
            source="test.py",
            result=mock_result,
            error=None,
        )
        assert result.index == 0
        assert result.source == "test.py"
        assert result.result is not None
        assert result.error is None

    def test_create_error_result(self) -> None:
        """Creates error MultiSourceResult."""
        result = MultiSourceResult(
            index=1,
            source="bad.py",
            result=None,
            error="Execution timed out",
        )
        assert result.index == 1
        assert result.source == "bad.py"
        assert result.result is None
        assert result.error == "Execution timed out"


class TestFormatMultiResultJson:
    """Tests for format_multi_result_json function."""

    def test_format_success_results(self, mock_result: ExecutionResult) -> None:
        """Formats successful results as JSON array."""
        import json

        results = [
            MultiSourceResult(index=0, source="test1.py", result=mock_result, error=None),
            MultiSourceResult(index=1, source="test2.py", result=mock_result, error=None),
        ]
        output = format_multi_result_json(results)
        parsed = json.loads(output)

        assert len(parsed) == 2
        assert parsed[0]["index"] == 0
        assert parsed[0]["source"] == "test1.py"
        assert parsed[0]["exit_code"] == 0
        assert parsed[1]["index"] == 1
        assert parsed[1]["source"] == "test2.py"

    def test_format_mixed_results(self, mock_result: ExecutionResult) -> None:
        """Formats mixed success and error results."""
        import json

        results = [
            MultiSourceResult(index=0, source="good.py", result=mock_result, error=None),
            MultiSourceResult(index=1, source="bad.py", result=None, error="Timed out"),
        ]
        output = format_multi_result_json(results)
        parsed = json.loads(output)

        assert len(parsed) == 2
        assert parsed[0]["exit_code"] == 0
        assert parsed[1]["error"] == "Timed out"
        assert parsed[1]["exit_code"] == EXIT_SANDBOX_ERROR


class TestCliMultiInput:
    """Tests for CLI multi-input functionality."""

    def test_multiple_inline_codes(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Executes multiple inline codes."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["print(1)", "print(2)"])

            assert result.exit_code == 0
            # Should have been called twice
            assert mock_scheduler.run.call_count == 2

    def test_multiple_c_flags(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Executes multiple -c flag codes."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["-c", "print(1)", "-c", "print(2)"])

            assert result.exit_code == 0
            assert mock_scheduler.run.call_count == 2

    def test_mixed_c_and_positional(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Executes mixed -c flags and positional sources."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["-c", "print(1)", "print(2)"])

            assert result.exit_code == 0
            # -c comes first, then positional
            assert mock_scheduler.run.call_count == 2

    def test_multiple_files(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """Executes multiple file sources."""
        # Create test files
        file1 = tmp_path / "test1.py"
        file1.write_text("print('file1')")
        file2 = tmp_path / "test2.py"
        file2.write_text("print('file2')")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, [str(file1), str(file2)])

            assert result.exit_code == 0
            assert mock_scheduler.run.call_count == 2

    def test_json_output_multi(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """JSON output for multiple sources is an array."""
        import json

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["--json", "print(1)", "print(2)"])

            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert isinstance(parsed, list)
            assert len(parsed) == 2
            assert all("index" in item for item in parsed)
            assert all("source" in item for item in parsed)

    def test_stdin_only_once(self, runner: CliRunner) -> None:
        """Stdin marker can only be used once."""
        result = runner.invoke(run_command, ["-", "-"], input="print(1)")
        assert result.exit_code == EXIT_CLI_ERROR
        assert "once" in result.output.lower()

    def test_single_source_uses_streaming(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Single source uses streaming (run_code) not run_multiple."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["print('hello')"])

            assert result.exit_code == 0


class TestCliMultiInputLanguageDetection:
    """Tests for language auto-detection with multiple sources."""

    def test_mixed_language_auto_detect(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """Auto-detects language per file extension."""
        py_file = tmp_path / "test.py"
        py_file.write_text("print(1)")
        js_file = tmp_path / "test.js"
        js_file.write_text("console.log(1)")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, [str(py_file), str(js_file)])

            assert result.exit_code == 0
            calls = mock_scheduler.run.call_args_list
            # Both languages should be detected (order may vary due to async execution)
            languages = [call.kwargs["language"] for call in calls]
            assert Language.PYTHON in languages
            assert Language.JAVASCRIPT in languages

    def test_global_language_override(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """Global -l flag overrides auto-detection for all sources."""
        py_file = tmp_path / "test.py"
        py_file.write_text("print(1)")
        js_file = tmp_path / "test.js"
        js_file.write_text("console.log(1)")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["-l", "raw", str(py_file), str(js_file)])

            assert result.exit_code == 0
            calls = mock_scheduler.run.call_args_list
            # Both should be raw (forced by -l)
            assert calls[0].kwargs["language"] == Language.RAW
            assert calls[1].kwargs["language"] == Language.RAW


class TestComputeMultiExitCode:
    """Tests for _compute_multi_exit_code helper."""

    def test_all_success(self, mock_result: ExecutionResult) -> None:
        """Returns SUCCESS when all sources succeed."""
        results = [
            MultiSourceResult(index=0, source="a", result=mock_result, error=None),
            MultiSourceResult(index=1, source="b", result=mock_result, error=None),
        ]
        assert _compute_multi_exit_code(results) == EXIT_SUCCESS

    def test_max_exit_code(self) -> None:
        """Returns max exit code from results."""
        result0 = ExecutionResult(
            stdout="",
            stderr="",
            exit_code=0,
            execution_time_ms=10,
            timing=TimingBreakdown(setup_ms=1, boot_ms=10, execute_ms=5, total_ms=16),
            warm_pool_hit=False,
        )
        result5 = ExecutionResult(
            stdout="",
            stderr="",
            exit_code=5,
            execution_time_ms=10,
            timing=TimingBreakdown(setup_ms=1, boot_ms=10, execute_ms=5, total_ms=16),
            warm_pool_hit=False,
        )
        results = [
            MultiSourceResult(index=0, source="a", result=result0, error=None),
            MultiSourceResult(index=1, source="b", result=result5, error=None),
        ]
        assert _compute_multi_exit_code(results) == 5

    def test_timeout_takes_precedence(self, mock_result: ExecutionResult) -> None:
        """Returns EXIT_TIMEOUT if any source timed out."""
        results = [
            MultiSourceResult(index=0, source="a", result=mock_result, error=None),
            MultiSourceResult(index=1, source="b", result=None, error="Execution timed out"),
        ]
        assert _compute_multi_exit_code(results) == EXIT_TIMEOUT

    def test_sandbox_error_returns_125(self) -> None:
        """Returns EXIT_SANDBOX_ERROR for non-timeout errors."""
        results = [
            MultiSourceResult(index=0, source="a", result=None, error="VM boot failed"),
        ]
        assert _compute_multi_exit_code(results) == EXIT_SANDBOX_ERROR

    def test_empty_results(self) -> None:
        """Returns SUCCESS for empty results list."""
        assert _compute_multi_exit_code([]) == EXIT_SUCCESS


class TestCliMultiInputExitCodes:
    """Tests for exit code handling with multiple sources."""

    def test_max_exit_code_returned(self, runner: CliRunner) -> None:
        """Returns maximum exit code from all sources."""
        result0 = ExecutionResult(
            stdout="",
            stderr="",
            exit_code=0,
            execution_time_ms=10,
            timing=TimingBreakdown(setup_ms=1, boot_ms=10, execute_ms=5, total_ms=16),
            warm_pool_hit=False,
        )
        result1 = ExecutionResult(
            stdout="",
            stderr="",
            exit_code=5,
            execution_time_ms=10,
            timing=TimingBreakdown(setup_ms=1, boot_ms=10, execute_ms=5, total_ms=16),
            warm_pool_hit=False,
        )

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.side_effect = [result0, result1]
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["print(1)", "exit(5)"])

            # Should return max(0, 5) = 5
            assert result.exit_code == 5

    def test_timeout_returns_124(self, runner: CliRunner) -> None:
        """Returns EXIT_TIMEOUT (124) if any source times out."""
        from exec_sandbox import VmTimeoutError

        result0 = ExecutionResult(
            stdout="",
            stderr="",
            exit_code=0,
            execution_time_ms=10,
            timing=TimingBreakdown(setup_ms=1, boot_ms=10, execute_ms=5, total_ms=16),
            warm_pool_hit=False,
        )

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            # First succeeds, second times out
            mock_scheduler.run.side_effect = [result0, VmTimeoutError("timeout")]
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(run_command, ["print(1)", "import time; time.sleep(100)"])

            assert result.exit_code == EXIT_TIMEOUT


# ============================================================================
# Prefetch Command Tests
# ============================================================================
class TestCliPrefetch:
    """Tests for CLI prefetch command."""

    def test_prefetch_help(self, runner: CliRunner) -> None:
        """Shows help for prefetch command."""
        result = runner.invoke(prefetch_command, ["--help"])
        assert result.exit_code == 0
        assert "Pre-download VM assets" in result.output
        assert "--arch" in result.output
        assert "--quiet" in result.output

    def test_prefetch_downloads_all(self, runner: CliRunner, tmp_path: Path) -> None:
        """Prefetch downloads all assets."""
        with patch("exec_sandbox.cli.prefetch_all_assets") as mock_prefetch:
            mock_prefetch.return_value = PrefetchResult(
                success=True,
                arch="aarch64",
                downloaded=["kernel", "initramfs"],
                cache_dir=tmp_path,
            )

            result = runner.invoke(prefetch_command, [])

            assert result.exit_code == 0
            mock_prefetch.assert_called_once_with(arch=None)

    def test_prefetch_with_arch(self, runner: CliRunner, tmp_path: Path) -> None:
        """Prefetch respects --arch flag."""
        with patch("exec_sandbox.cli.prefetch_all_assets") as mock_prefetch:
            mock_prefetch.return_value = PrefetchResult(
                success=True,
                arch="aarch64",
                downloaded=["kernel", "initramfs"],
                cache_dir=tmp_path,
            )

            result = runner.invoke(prefetch_command, ["--arch", "aarch64"])

            assert result.exit_code == 0
            mock_prefetch.assert_called_once_with(arch="aarch64")

    def test_prefetch_quiet(self, runner: CliRunner, tmp_path: Path) -> None:
        """Prefetch respects -q flag (suppresses output on success)."""
        with patch("exec_sandbox.cli.prefetch_all_assets") as mock_prefetch:
            mock_prefetch.return_value = PrefetchResult(
                success=True,
                arch="aarch64",
                downloaded=["kernel", "initramfs"],
                cache_dir=tmp_path,
            )

            result = runner.invoke(prefetch_command, ["-q"])

            assert result.exit_code == 0
            # Quiet mode should produce no output on success
            assert result.output == ""

    def test_prefetch_returns_error_on_failure(self, runner: CliRunner) -> None:
        """Prefetch returns error code on failure."""
        with patch("exec_sandbox.cli.prefetch_all_assets") as mock_prefetch:
            mock_prefetch.return_value = PrefetchResult(
                success=False,
                arch="aarch64",
                errors=[("kernel", "Network error")],
            )

            result = runner.invoke(prefetch_command, [])

            assert result.exit_code == 1


# ============================================================================
# CLI Group Tests (Top-level help and version)
# ============================================================================
class TestCliGroup:
    """Tests for CLI group (top-level commands)."""

    def test_cli_help(self, runner: CliRunner) -> None:
        """Shows help for main CLI group."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "exec-sandbox" in result.output
        assert "run" in result.output
        assert "prefetch" in result.output

    def test_cli_version(self, runner: CliRunner) -> None:
        """Shows version with --version flag."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_cli_short_version(self, runner: CliRunner) -> None:
        """Shows version with -V flag."""
        result = runner.invoke(cli, ["-V"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_run_subcommand_help(self, runner: CliRunner) -> None:
        """Shows help for run subcommand."""
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "Execute code" in result.output
        assert "--language" in result.output

    def test_prefetch_subcommand_help(self, runner: CliRunner) -> None:
        """Shows help for prefetch subcommand."""
        result = runner.invoke(cli, ["prefetch", "--help"])
        assert result.exit_code == 0
        assert "Pre-download VM assets" in result.output


# ============================================================================
# Backward Compatibility Tests
# ============================================================================
class TestCliBackwardCompat:
    """Tests for CLI backward compatibility (implicit run insertion)."""

    def test_explicit_run(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Explicit 'run' subcommand works."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = AsyncMock()
            mock_scheduler.run.return_value = mock_result
            mock_scheduler.__aenter__.return_value = mock_scheduler
            mock_scheduler.__aexit__.return_value = None
            scheduler_cls.return_value = mock_scheduler

            result = runner.invoke(cli, ["run", "print(1)"])

            assert result.exit_code == 0
            mock_scheduler.run.assert_called_once()

    def test_explicit_prefetch(self, runner: CliRunner, tmp_path: Path) -> None:
        """Explicit 'prefetch' subcommand works."""
        with patch("exec_sandbox.cli.prefetch_all_assets") as mock_prefetch:
            mock_prefetch.return_value = PrefetchResult(
                success=True,
                arch="aarch64",
                downloaded=["kernel"],
                cache_dir=tmp_path,
            )

            result = runner.invoke(cli, ["prefetch"])

            assert result.exit_code == 0
            mock_prefetch.assert_called_once()
