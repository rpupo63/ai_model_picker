#!/bin/bash
cd "$(dirname "$0")"
rm -rf dist build pkg src *.egg-info
makepkg -si
