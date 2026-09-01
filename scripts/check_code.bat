@echo off
REM DNALLM Code Quality Check Script for Windows
REM This script runs all code quality checks required before committing
REM Usage: scripts\check_code.bat [--fix] [--verbose]

setlocal enabledelayedexpansion

REM Default options
set FIX_MODE=false
set VERBOSE=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse
if "%~1"=="--fix" (
    set FIX_MODE=true
    shift
    goto :parse_args
)
if "%~1"=="--verbose" (
    set VERBOSE=true
    shift
    goto :parse_args
)
if "%~1"=="-h" goto :show_help
if "%~1"=="--help" goto :show_help
echo Unknown option: %~1
echo Use -h or --help for usage information
exit /b 1

:show_help
echo Usage: %0 [--fix] [--verbose]
echo   --fix     Auto-fix issues where possible
echo   --verbose Show detailed output
echo   -h, --help Show this help message
exit /b 0

:end_parse

REM Check if we're in the right directory
if not exist "pyproject.toml" (
    echo [ERROR] Please run this script from the DNALLM root directory
    exit /b 1
)

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo [WARNING] Virtual environment not detected. Please activate it first:
    echo   .venv\Scripts\activate
    echo.
    set /p CONTINUE="Continue anyway? (y/N): "
    if /i not "!CONTINUE!"=="y" exit /b 1
)

echo [INFO] Starting DNALLM code quality checks...
echo ==========================================

REM Track overall success
set OVERALL_SUCCESS=true

REM 1. Code Formatting Check
echo [INFO] 1. Checking code formatting with Ruff...
if "%FIX_MODE%"=="true" (
    call ruff format .
    if !errorlevel! neq 0 (
        echo [ERROR] Code formatting failed
        set OVERALL_SUCCESS=false
    ) else (
        echo [SUCCESS] Code formatting completed successfully
    )
) else (
    call ruff format --check .
    if !errorlevel! neq 0 (
        echo [WARNING] Code formatting issues found. Run with --fix to auto-fix.
        set OVERALL_SUCCESS=false
    ) else (
        echo [SUCCESS] Code formatting check passed
    )
)
echo.

REM 2. Code Quality Check with Ruff
echo [INFO] 2. Running code quality checks with Ruff...
if "%FIX_MODE%"=="true" (
    call ruff check . --fix --statistics
    if !errorlevel! neq 0 (
        echo [ERROR] Code quality check failed
        set OVERALL_SUCCESS=false
    ) else (
        echo [SUCCESS] Code quality check completed successfully
    )
) else (
    call ruff check . --statistics
    if !errorlevel! neq 0 (
        echo [WARNING] Code quality issues found. Run with --fix to auto-fix.
        set OVERALL_SUCCESS=false
    ) else (
        echo [SUCCESS] Code quality check passed
    )
)
echo.

REM 3. Flake8 Check for MCP Module
echo [INFO] 3. Running flake8 check for MCP module...
call flake8 dnallm/mcp/ --max-line-length=79 --extend-ignore=E203,W503,C901,E402
if !errorlevel! neq 0 (
    echo [ERROR] Flake8 check failed
    set OVERALL_SUCCESS=false
) else (
    echo [SUCCESS] Flake8 check passed
)
echo.

REM 4. Type Checking with MyPy
echo [INFO] 4. Running type checking with MyPy...
call mypy dnallm/ --ignore-missing-imports --no-strict-optional --disable-error-code=var-annotated --disable-error-code=assignment --disable-error-code=return-value --disable-error-code=arg-type --disable-error-code=index --disable-error-code=attr-defined --disable-error-code=operator --disable-error-code=call-overload --disable-error-code=valid-type --disable-error-code=no-redef --disable-error-code=dict-item --disable-error-code=return --disable-error-code=unreachable --disable-error-code=misc --disable-error-code=import-untyped
if !errorlevel! neq 0 (
    echo [ERROR] Type checking failed
    set OVERALL_SUCCESS=false
) else (
    echo [SUCCESS] Type checking passed
)
echo.

REM 5. Test Suite
echo [INFO] 5. Running test suite...
call pytest tests/ -v --tb=short
if !errorlevel! neq 0 (
    echo [ERROR] Test suite failed
    set OVERALL_SUCCESS=false
) else (
    echo [SUCCESS] Test suite passed
)
echo.

REM 6. Test Coverage (only if not in fix mode)
if "%FIX_MODE%"=="false" (
    echo [INFO] 6. Running test coverage analysis...
    call pytest tests/ --cov=dnallm --cov-report=term-missing --cov-report=xml
    if !errorlevel! neq 0 (
        echo [ERROR] Test coverage analysis failed
        set OVERALL_SUCCESS=false
    ) else (
        echo [SUCCESS] Test coverage analysis passed
    )
    echo.
)

REM Final Results
echo ==========================================
if "%OVERALL_SUCCESS%"=="true" (
    echo [SUCCESS] All checks passed! ✅
    echo [INFO] Your code is ready for commit.
    exit /b 0
) else (
    echo [ERROR] Some checks failed! ❌
    echo [INFO] Please fix the issues above before committing.
    if "%FIX_MODE%"=="false" (
        echo [INFO] Run with --fix to auto-fix some issues: scripts\check_code.bat --fix
    )
    exit /b 1
)
