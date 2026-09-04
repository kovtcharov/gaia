# Process image names that `gaia kill --port` / `gaia api stop` may terminate.
# Matched case-insensitively as a substring, so `gaia.exe`, `lemonade-server`
# and the venv `python.exe` running the API server all qualify while an
# unrelated program listening on a mistyped port does not. Same identity check
# `LemonadeEmbedded._daemon_alive` makes before trusting a recorded pid.
_KILLABLE_PROCESS_NAMES = (
    "gaia",
    "lemonade",
    "lemond",
    "python",
    "pythonw",
    "node",
    "electron",
)


def _address_port(address):
    """Return the port field of a netstat address (``0.0.0.0:80``, ``[::]:80``)."""
    return address.rsplit(":", 1)[-1] if ":" in address else ""


def _is_listening_row(state, foreign_address):
    """Whether a netstat TCP row is a listening socket rather than a connection.

    The state column is localized on non-English Windows, so a listener is also
    recognised structurally: only a listening socket has no foreign port.
    """
    if state.upper() in ("LISTENING", "LISTEN"):
        return True
    return _address_port(foreign_address) in ("0", "*")


def _parse_windows_netstat_listeners(output, port):
    """PIDs of TCP sockets listening on exactly ``port`` in ``netstat -ano`` output.

    Columns are parsed rather than substring-matched: ``":80" in line`` also
    matches a ``:8009`` foreign address and every ESTABLISHED / TIME_WAIT row,
    which is how a mistyped port used to kill an unrelated process.
    """
    pids = []
    for line in output.splitlines():
        parts = line.split()
        # A TCP row is exactly: proto, local, foreign, state, pid. UDP rows
        # have no state column and are never what a server is listening on.
        if len(parts) != 5 or not parts[0].upper().startswith("TCP"):
            continue
        _, local, foreign, state, pid_field = parts
        if _address_port(local) != str(port):
            continue
        if not _is_listening_row(state, foreign):
            continue
        try:
            pid = int(pid_field)
        except ValueError:
            continue
        if pid > 0 and pid not in pids:
            pids.append(pid)
    return pids


def _parse_unix_netstat_listeners(output, port):
    """``(pid, process_name)`` pairs listening on ``port`` in ``netstat -tulpn`` output."""
    found = []
    for line in output.splitlines():
        parts = line.split()
        # proto, recv-q, send-q, local, foreign, state, pid/program
        if len(parts) < 7 or not parts[0].lower().startswith("tcp"):
            continue
        local, foreign, state, program = parts[3], parts[4], parts[5], parts[6]
        if _address_port(local) != str(port):
            continue
        if not _is_listening_row(state, foreign):
            continue
        pid_field, _, name = program.partition("/")
        try:
            pid = int(pid_field)
        except ValueError:
            continue
        if pid > 0 and pid not in [p for p, _ in found]:
            found.append((pid, name.strip()))
    return found


def _process_image_name(pid):
    """Best-effort lowercased image name of ``pid``; empty string when unknown."""
    try:
        if sys.platform.startswith("win"):
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=15,
                check=False,
            )
            first = result.stdout.strip().splitlines()[:1]
            if not first or not first[0].startswith('"'):
                return ""
            return first[0].split('","')[0].strip('"').lower()
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
            check=False,
        )
        return result.stdout.strip().lower()
    except (OSError, subprocess.SubprocessError) as e:
        log.warning(f"Could not read the process name for pid {pid}: {e}")
        return ""


def _is_killable_process(name):
    """Whether a process image name belongs to GAIA or Lemonade."""
    lowered = (name or "").lower()
    return any(allowed in lowered for allowed in _KILLABLE_PROCESS_NAMES)


def _listeners_on_port(port):
    """Return ``(pid, process_name)`` for every process listening on ``port``."""
    if sys.platform.startswith("win"):
        output = subprocess.check_output(
            ["netstat", "-ano"], text=True, errors="replace"
        )
        return [
            (pid, _process_image_name(pid))
            for pid in _parse_windows_netstat_listeners(output, port)
        ]

    try:
        # -sTCP:LISTEN keeps the client end of every connection out of the
        # result; without it lsof returns both ends and kill -9 took out the
        # Agent UI backend and any `gaia chat` talking to Lemonade.
        output = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            text=True,
            errors="replace",
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        output = None

    if output is not None:
        listeners = []
        for entry in output.split():
            try:
                pid = int(entry.strip())
            except ValueError:
                continue
            if pid > 0:
                listeners.append((pid, _process_image_name(pid)))
        return listeners

    # lsof is unavailable: fall back to netstat, which names the process itself.
    netstat_output = subprocess.check_output(
        ["netstat", "-tulpn"], text=True, errors="replace"
    )
    return _parse_unix_netstat_listeners(netstat_output, port)


def _terminate_pid(pid):
    """Terminate ``pid`` with the platform's forceful kill."""
    if sys.platform.startswith("win"):
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], shell=False, check=True)
    else:
        subprocess.run(["kill", "-9", str(pid)], shell=False, check=True)


def kill_process_by_port(port):
    """Kill the GAIA/Lemonade process listening on ``port``.

    Only sockets in the LISTENING state whose *local* port is exactly ``port``
    count, and only processes whose image name is GAIA's or Lemonade's are
    terminated — a mistyped port reports a refusal instead of killing whatever
    else happens to be running.
    """
    try:
        port = int(port)
    except (ValueError, TypeError):
        return {"success": False, "message": f"Invalid port number: {port!r}"}

    try:
        listeners = _listeners_on_port(port)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "success": False,
            "message": f"No process found listening on port {port}",
        }
    except (OSError, subprocess.SubprocessError) as e:
        return {"success": False, "message": f"Could not inspect port {port}: {e}"}

    if not listeners:
        return {"success": False, "message": f"No process is listening on port {port}"}

    killed = []
    refused = []
    failed = []
    for pid, name in listeners:
        if not _is_killable_process(name):
            refused.append(f"{pid} ({name or 'unknown process'})")
            continue
        try:
            _terminate_pid(pid)
            killed.append(str(pid))
        except (subprocess.CalledProcessError, OSError) as e:
            failed.append(f"{pid}: {e}")

    if killed:
        return {
            "success": True,
            "message": f"Killed process(es) {', '.join(killed)} listening on port {port}",
        }

    if refused:
        return {
            "success": False,
            "message": (
                f"Refusing to kill {', '.join(refused)} on port {port}: not a "
                f"GAIA or Lemonade process. Stop it with its own tooling, or "
                f"kill it by PID if that is really what you want."
            ),
        }

    return {
        "success": False,
        "message": f"Failed to kill the process on port {port} ({'; '.join(failed)})",
    }
