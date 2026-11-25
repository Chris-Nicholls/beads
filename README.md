Velvet reverb firmware for Beads hardware.

A reverb-only firmware for the Beads module: the DLD's velvet-noise reverb
(`beads/velvet/`) in place of the granular engine, with the panel remapped
to the reverb's controls (see the panel notes at the top of
`beads/beads.cc`). The original granular main is kept as
`beads/beads_granular.cc.orig`.

* `beads/` – the firmware, its drivers, resources and bootloader.
* `beads/velvet/` – the reverb engine and the wet-signal effects.
* `stmlib/`, `stm_audio_bootloader/` – support libraries (submodules).

Building
========

    git submodule update --init
    make -f beads/makefile bin

Set `TOOLCHAIN_PATH` to the folder holding `bin/arm-none-eabi-gcc` if it is
not the makefile's default (for example `TOOLCHAIN_PATH=/opt/homebrew/`).

Code provided as is, not verified to run on every module. Install at your own risk.

License
=======

Code: MIT license.

Hardware: cc-by-sa-3.0

Original Beads code and hardware by Emilie Gillet (emilie.o.gillet@gmail.com).

Guidelines for derivative works
===============================

**Mutable Instruments is a registered trademark.**

The name "Mutable Instruments" should not be used on any of the derivative works you create from these files.

We do not recommend you to keep the original name of the Mutable Instruments module for your derivative works.
