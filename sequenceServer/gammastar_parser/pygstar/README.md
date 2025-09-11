# Python Wrapper for the [GammaStar backend](https://gitlab.fme.lan/hussain/gstar-parser) written in RUST
### Compatible with the [MRZeroCORE](https://github.com/MRsources/MRzero-Core) version 0.3.6
Serves as a gammaStar interpreter to simulate arbitrary GammaStar Sequences on MRZero simulation framework.

## Building from source

This assumes windows as host operating system. For building the python wheel, you need:
- the Rust toolchain: [rustup](https://rustup.rs/)
- the rust-python build tool tool: [pip install maturin](https://github.com/PyO3/maturin)

**Building**

```
maturin build --interpreter python
```

 ## Cross-Compilation while on WSL
To cross-compile a Rust-based Python extension module for different architectures using `maturin`, follow these steps:

**1. Install the Cross-Compilation Toolchain**

Ensure that the necessary cross-compilation tools are installed on your system. The specific toolchain depends on the target architecture:

- **For Windows (x86_64)**:

  On Debian-based systems, install the MinGW-w64 toolchain:

  ```sh
  sudo apt install g++-mingw-w64-x86-64
  ```

- **For ARM (e.g., Raspberry Pi)**:

  Install the ARM cross-compilation tools:

  ```sh
  sudo apt install g++-aarch64-linux-gnu
  ```

**2. Add the Rust Target for the Desired Architecture**

Use `rustup` to add the appropriate target for cross-compilation:

- **For Windows (x86_64)**:

  ```sh
  rustup target add x86_64-pc-windows-gnu
  ```

- **For ARM (e.g., Raspberry Pi)**:

  ```sh
  rustup target add aarch64-unknown-linux-gnu
  ```

**3. Configure the Linker in `.cargo/config.toml`**

Create or edit the `.cargo/config.toml` file in your project to specify the linker for the target architecture:

- **For Windows (x86_64)**:

  ```toml
  [target.x86_64-pc-windows-gnu]
  linker = "x86_64-w64-mingw32-gcc"
  ```

- **For ARM (e.g., Raspberry Pi)**:

  ```toml
  [target.aarch64-unknown-linux-gnu]
  linker = "aarch64-linux-gnu-gcc"
  ```

**4. Build the Python Extension Module with `maturin`**

Use `maturin` to build the Python extension module, specifying the Python interpreter version and the target architecture:

- **For Windows (x86_64)**:

  ```sh
  maturin build --interpreter python3.xx --target x86_64-pc-windows-gnu
  ```

- **For ARM (e.g., Raspberry Pi)**:

  ```sh
  maturin build --interpreter python3.xx --target aarch64-unknown-linux-gnu
  ```

Replace `python3.xx` with the specific Python version you are targeting (e.g., `python3.10`).

**5. Verify the Generated Wheel**

After the build completes, verify that the wheel file has been created in the `target/wheels` directory. The filename should indicate compatibility with the specified Python version and target platform, such as:

- **For Windows (x86_64)**:

  ```
  your_package_name-0.1.0-py3-none-win_amd64.whl
  ```

- **For ARM (e.g., Raspberry Pi)**:

  ```
  your_package_name-0.1.0-py3-none-manylinux2014_aarch64.whl
  ```

**6. Test the Wheel on the Target System**

Transfer the generated wheel file to the target system and install it using `pip` to ensure it functions as expected:

```sh
pip install your_package_name-0.1.0-py3-none-<platform>.whl
```

Replace `<platform>` with the appropriate platform tag (e.g., `win_amd64` for Windows or `manylinux2014_aarch64` for ARM).

**Additional Considerations**

- **Python Version Compatibility**: Ensure that the Python interpreter specified with the `--interpreter` flag matches the version you intend to target on the destination system.

- **Dependencies**: If your Rust project has dependencies that require additional configuration for cross-compilation, ensure those are properly addressed in your setup.

- **Testing**: It's advisable to test the built wheel in a clean environment on the target architecture to confirm that all dependencies are correctly packaged and that the module functions as intended.

By following these steps and adjusting the toolchain, Rust target, linker configuration, and build commands accordingly, you can cross-compile your Python extension module for various architectures using `maturin`. 