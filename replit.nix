{ pkgs }:
{
  deps = [
    pkgs.git
    pkgs.zlib
    pkgs.pkg-config
    pkgs.cmake
    pkgs.gnumake
    pkgs.gcc
   pkgs.nasm ];
}