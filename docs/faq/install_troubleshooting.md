# Installation Troubleshooting

Please see [getting_started](../getting_started/installation.md) section for detailed usage, any troubleshooting not recorded will be added here.

#### `uv pip install` fails with "Read-only file system (os error 30)"

**Error:**
```
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation --link-mode=copy
error: Read-only file system (os error 30) at path "/tmp/.tmpEqRyCv"
```

**Cause:**
The `/tmp` directory or its partition is mounted as read-only, which is common in certain container environments, shared servers, or security-hardened systems. Installation requires writing temporary files, so it fails.

**Solution:**
You can resolve this by setting the `TMPDIR` environment variable to point to a writable directory (such as your current working directory or `/var/tmp`):

```bash
export TMPDIR=/path/to/writable/directory
# Example: use current directory
export TMPDIR=$(pwd)/tmp
mkdir -p $TMPDIR
```

Then retry your installation command.
