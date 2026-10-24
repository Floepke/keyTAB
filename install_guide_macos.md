# Set up and build keyTAB on macOS

Follow these steps to run keyTAB and create a Mac app.
You need an internet connection.

**Important:** on the maintainer's Mac, PySide6 only works when installed
locally, **without a virtual environment (venv)**. This guide uses that setup.
Do not create or activate a venv.

## 1. Install the tools

1. Download **Python 3.13** using the **macOS installer** from
   [python.org](https://www.python.org/downloads/macos/) and install it.

2. Open **Terminal** from Applications > Utilities. Paste this command and
   press **Return**:

   ```sh
   xcode-select --install
   ```

   Finish the installation window before continuing. If the tools are already
   installed, you can continue.

3. Install **Homebrew** by following the instructions at [brew.sh](https://brew.sh/).
   Also follow the **Next steps** shown at the end of its installation.

4. Close Terminal, open it again, and run:

   ```sh
   brew install cairo pkg-config
   ```

For all commands below, paste each line into Terminal, press **Return**, and
wait until it finishes before entering the next one. Keep this Terminal window
open for the remaining steps. If a command shows an error, stop rather than
continuing.

## 2. Download keyTAB

These commands download the project into your Documents folder:

```sh
cd "$HOME/Documents"
git clone https://github.com/Floepke/keyTAB.git
cd keyTAB
```

**Already have keyTAB in Documents?** Skip the commands above and run only:

```sh
cd "$HOME/Documents/keyTAB"
```

## 3. Install the Python packages

First, select the Python you installed in step 1:

```sh
PYTHON="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13"
```

Then install the packages keyTAB needs. This may take a few minutes:

```sh
"$PYTHON" -m pip install --user --upgrade pip setuptools wheel
"$PYTHON" -m pip install --user -r requirements.txt
"$PYTHON" -m pip install --user pyinstaller
```

The `--user` option installs packages locally for your Mac user, without a
venv. Do not use `sudo` for these commands.

## 4. Try keyTAB

```sh
"$PYTHON" keyTAB.py
```

The keyTAB window should open. Try opening a score from the
[`example .piano files`](example%20.piano%20files/) folder.
Close keyTAB before continuing.

## 5. Build the Mac app

Check **Apple menu > About This Mac**:

* If it shows an **Apple M-series chip**, run:

  ```sh
  "$PYTHON" deploy/build_.app.py --target-arch arm64
  ```

* If it shows an **Intel processor**, run:

  ```sh
  "$PYTHON" deploy/build_.app.py --target-arch x86_64
  ```

Wait for the build to finish. It can take several minutes.
The script creates the app and puts an installer on your **Desktop**.
It replaces any previous installer with the same name.

### Universal2 maintainer build

To distribute one `.app` that runs natively on both Apple Silicon and Intel
Macs, build each architecture from its own virtual environment, then let the
build script merge them. This is required because compiled packages such as
NumPy and python-rtmidi are architecture-specific.

On an Apple Silicon Mac, install Rosetta first:

```sh
softwareupdate --install-rosetta --agree-to-license
```

With `PYTHON` still set to the universal Python.org interpreter from step 3,
create and populate both environments:

```sh
arch -arm64 "$PYTHON" -m venv .venv-arm64
arch -x86_64 "$PYTHON" -m venv .venv-x86_64
arch -arm64 .venv-arm64/bin/python -m pip install --upgrade pip setuptools wheel
arch -x86_64 .venv-x86_64/bin/python -m pip install --upgrade pip setuptools wheel
arch -arm64 .venv-arm64/bin/python -m pip install -r requirements.txt pyinstaller
arch -x86_64 .venv-x86_64/bin/python -m pip install -r requirements.txt pyinstaller
```

Then build the universal bundle:

```sh
"$PYTHON" deploy/build_.app.py --target-arch universal2
```

The default paths are `.venv-arm64/bin/python` and
`.venv-x86_64/bin/python`. Supply `--arm64-python` or `--x86-64-python` to
use environments stored elsewhere. The script builds two thin bundles, merges
their shared Mach-O executables and libraries with `lipo`, and verifies that
each shared binary contains both `arm64` and `x86_64` slices. It retains a
single-architecture support library only when the other architecture does not
link to it.

## 6. Install the app

1. On your Desktop, open **keyTAB-1.1.4.dmg**. The version number may differ.
2. Drag **keyTAB** onto **Applications** in the window that opens.
3. Eject the disk image, then open keyTAB from **Applications**.

The app is inside the DMG; it is normal not to see a separate app on the
Desktop. If creating the DMG failed but the app built successfully, the script
leaves a **keyTAB-version.app** on your Desktop instead. Drag that to Applications.

## If something goes wrong

* **macOS blocks the app:** if you trust the build, look in **System Settings >
  Privacy & Security** for **Open Anyway**. This build process does not include
  Apple's signing and notarization steps for public distribution.

* **`brew`** **is not found:** complete Homebrew's **Next steps**, then reopen
  Terminal.

* **Python or a package is not found:** repeat step 3. If you reopened Terminal,
  first run `cd "$HOME/Documents/keyTAB"` again.

* **Another error:** copy the error text from Terminal when asking for help.
  If a build failed, its temporary files are kept in **Desktop > keyTAB\_build**
  until the next build.
