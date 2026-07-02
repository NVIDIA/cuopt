/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <file_to_string.hpp>

#include <utilities/error.hpp>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#ifdef MPS_PARSER_WITH_BZIP2
#include <bzlib.h>
#endif  // MPS_PARSER_WITH_BZIP2

#ifdef MPS_PARSER_WITH_ZLIB
#include <zlib.h>
#endif  // MPS_PARSER_WITH_ZLIB

#if defined(MPS_PARSER_WITH_BZIP2) || defined(MPS_PARSER_WITH_ZLIB)
#include <dlfcn.h>
#endif  // MPS_PARSER_WITH_BZIP2 || MPS_PARSER_WITH_ZLIB

namespace {
using cuopt::mathematical_optimization::io::error_type_t;
using cuopt::mathematical_optimization::io::mps_parser_expects;
using cuopt::mathematical_optimization::io::mps_parser_expects_fatal;

struct FcloseDeleter {
  void operator()(FILE* fp)
  {
    mps_parser_expects_fatal(
      fclose(fp) == 0, error_type_t::ValidationError, "Error closing input file!");
  }
};
}  // end namespace

#ifdef MPS_PARSER_WITH_BZIP2
namespace {
using BZ2_bzReadOpen_t  = decltype(&BZ2_bzReadOpen);
using BZ2_bzReadClose_t = decltype(&BZ2_bzReadClose);
using BZ2_bzRead_t      = decltype(&BZ2_bzRead);

std::vector<char> bz2_file_to_string(const std::string& file)
{
  struct DlCloseDeleter {
    void operator()(void* fp)
    {
      mps_parser_expects_fatal(
        dlclose(fp) == 0, error_type_t::ValidationError, "Error closing libbz2.so!");
    }
  };
  struct BzReadCloseDeleter {
    void operator()(void* f)
    {
      int bzerror;
      if (f != nullptr) fptr(&bzerror, f);
      mps_parser_expects_fatal(
        bzerror == BZ_OK, error_type_t::ValidationError, "Error closing bzip2 file!");
    }
    BZ2_bzReadClose_t fptr = nullptr;
  };

  // bzip2 upstream's stable SONAME is "libbz2.so.1.0". On most distros only the
  // versioned name(s) ship in the runtime package; the unversioned "libbz2.so"
  // symlink is dev-only. Try names in decreasing specificity.
  void* raw_lbz2handle = nullptr;
  for (const char* soname : {"libbz2.so.1.0", "libbz2.so.1", "libbz2.so"}) {
    raw_lbz2handle = dlopen(soname, RTLD_LAZY);
    if (raw_lbz2handle != nullptr) break;
  }
  std::unique_ptr<void, DlCloseDeleter> lbz2handle{raw_lbz2handle};
  mps_parser_expects(
    lbz2handle != nullptr,
    error_type_t::ValidationError,
    "Could not open .bz2 file since libbz2 was not found "
    "(tried libbz2.so.1.0, libbz2.so.1, libbz2.so). In order to open .bz2 files "
    "directly, please ensure libbzip2 is installed. Alternatively, decompress the .bz2 file "
    "manually and open the uncompressed file. Given path: %s",
    file.c_str());

  BZ2_bzReadOpen_t BZ2_bzReadOpen =
    reinterpret_cast<BZ2_bzReadOpen_t>(dlsym(lbz2handle.get(), "BZ2_bzReadOpen"));
  BZ2_bzReadClose_t BZ2_bzReadClose =
    reinterpret_cast<BZ2_bzReadClose_t>(dlsym(lbz2handle.get(), "BZ2_bzReadClose"));
  BZ2_bzRead_t BZ2_bzRead = reinterpret_cast<BZ2_bzRead_t>(dlsym(lbz2handle.get(), "BZ2_bzRead"));
  mps_parser_expects(
    BZ2_bzReadOpen != nullptr && BZ2_bzReadClose != nullptr && BZ2_bzRead != nullptr,
    error_type_t::ValidationError,
    "Error loading libbzip2! Library version might be incompatible. Please decompress the .bz2 "
    "file manually and open the uncompressed file. Given path: %s",
    file.c_str());

  std::unique_ptr<FILE, FcloseDeleter> fp{fopen(file.c_str(), "rb")};
  mps_parser_expects(fp != nullptr,
                     error_type_t::ValidationError,
                     "Error opening input file! Given path: %s",
                     file.c_str());
  int bzerror = BZ_OK;
  std::unique_ptr<void, BzReadCloseDeleter> bzfile{
    BZ2_bzReadOpen(&bzerror, fp.get(), 0, 0, nullptr, 0), {BZ2_bzReadClose}};
  mps_parser_expects(bzerror == BZ_OK,
                     error_type_t::ValidationError,
                     "Could not open bzip2 compressed file! Given path: %s",
                     file.c_str());

  std::vector<char> buf;
  const size_t readbufsize = 1ull << 24;  // 16MiB - just a guess.
  std::vector<char> readbuf(readbufsize);
  while (bzerror == BZ_OK) {
    const size_t bytes_read = BZ2_bzRead(&bzerror, bzfile.get(), readbuf.data(), readbuf.size());
    if (bzerror == BZ_OK || bzerror == BZ_STREAM_END) {
      buf.insert(buf.end(), begin(readbuf), begin(readbuf) + bytes_read);
    }
  }
  buf.push_back('\0');
  mps_parser_expects(bzerror == BZ_STREAM_END,
                     error_type_t::ValidationError,
                     "Error in bzip2 decompression of input file! Given path: %s",
                     file.c_str());
  return buf;
}
}  // end namespace
#endif  // MPS_PARSER_WITH_BZIP2

#ifdef MPS_PARSER_WITH_ZLIB
namespace {
using gzopen_t    = decltype(&gzopen);
using gzclose_r_t = decltype(&gzclose_r);
using gzbuffer_t  = decltype(&gzbuffer);
using gzread_t    = decltype(&gzread);
using gzerror_t   = decltype(&gzerror);

std::vector<char> zlib_file_to_string(const std::string& file)
{
  struct DlCloseDeleter {
    void operator()(void* fp)
    {
      mps_parser_expects_fatal(
        dlclose(fp) == 0, error_type_t::ValidationError, "Error closing libz.so!");
    }
  };
  struct GzCloseDeleter {
    void operator()(gzFile_s* f)
    {
      int err = fptr(f);
      mps_parser_expects_fatal(
        err == Z_OK, error_type_t::ValidationError, "Error closing gz file!");
    }
    gzclose_r_t fptr = nullptr;
  };

  std::unique_ptr<void, DlCloseDeleter> lzhandle{dlopen("libz.so.1", RTLD_LAZY)};
  mps_parser_expects(
    lzhandle != nullptr,
    error_type_t::ValidationError,
    "Could not open .gz file since libz.so was not found. In order to open .gz files "
    "directly, please ensure zlib is installed. Alternatively, decompress the .gz file "
    "manually and open the uncompressed file. Given path: %s",
    file.c_str());
  gzopen_t gzopen       = reinterpret_cast<gzopen_t>(dlsym(lzhandle.get(), "gzopen"));
  gzclose_r_t gzclose_r = reinterpret_cast<gzclose_r_t>(dlsym(lzhandle.get(), "gzclose_r"));
  gzbuffer_t gzbuffer   = reinterpret_cast<gzbuffer_t>(dlsym(lzhandle.get(), "gzbuffer"));
  gzread_t gzread       = reinterpret_cast<gzread_t>(dlsym(lzhandle.get(), "gzread"));
  gzerror_t gzerror     = reinterpret_cast<gzerror_t>(dlsym(lzhandle.get(), "gzerror"));
  mps_parser_expects(
    gzopen != nullptr && gzclose_r != nullptr && gzbuffer != nullptr && gzread != nullptr &&
      gzerror != nullptr,
    error_type_t::ValidationError,
    "Error loading zlib! Library version might be incompatible. Please decompress the .gz file "
    "manually and open the uncompressed file. Given path: %s",
    file.c_str());
  std::unique_ptr<gzFile_s, GzCloseDeleter> gzfp{gzopen(file.c_str(), "rb"), {gzclose_r}};
  mps_parser_expects(gzfp != nullptr,
                     error_type_t::ValidationError,
                     "Error opening compressed input file! Given path: %s",
                     file.c_str());
  int zlib_status = gzbuffer(gzfp.get(), 1 << 20);  // 1 MiB
  mps_parser_expects(zlib_status == Z_OK,
                     error_type_t::ValidationError,
                     "Could not set zlib internal buffer size for decompression! Given path: %s",
                     file.c_str());
  std::vector<char> buf;
  const size_t readbufsize = 1ull << 24;  // 16MiB
  std::vector<char> readbuf(readbufsize);
  int bytes_read = -1;
  while (bytes_read != 0) {
    bytes_read = gzread(gzfp.get(), readbuf.data(), readbuf.size());
    if (bytes_read > 0) { buf.insert(buf.end(), begin(readbuf), begin(readbuf) + bytes_read); }
    if (bytes_read < 0) {
      gzerror(gzfp.get(), &zlib_status);
      break;
    }
  }
  buf.push_back('\0');
  mps_parser_expects(zlib_status == Z_OK,
                     error_type_t::ValidationError,
                     "Error in zlib decompression of input file! Given path: %s",
                     file.c_str());
  return buf;
}
}  // end namespace
#endif  // MPS_PARSER_WITH_ZLIB

namespace cuopt::mathematical_optimization::io {

std::vector<char> file_to_string(const std::string& file)
{
#ifdef MPS_PARSER_WITH_BZIP2
  if (file.size() > 4 && file.substr(file.size() - 4, 4) == ".bz2") {
    return bz2_file_to_string(file);
  }
#endif  // MPS_PARSER_WITH_BZIP2

#ifdef MPS_PARSER_WITH_ZLIB
  if (file.size() > 3 && file.substr(file.size() - 3, 3) == ".gz") {
    return zlib_file_to_string(file);
  }
#endif  // MPS_PARSER_WITH_ZLIB

  // Faster than using C++ I/O
  std::unique_ptr<FILE, FcloseDeleter> fp{fopen(file.c_str(), "r")};
  mps_parser_expects(fp != nullptr,
                     error_type_t::ValidationError,
                     "Error opening input file! Given path: %s",
                     file.c_str());

  mps_parser_expects(fseek(fp.get(), 0L, SEEK_END) == 0,
                     error_type_t::ValidationError,
                     "Error seeking input file! Given path: %s",
                     file.c_str());
  const long bufsize = ftell(fp.get());
  mps_parser_expects(bufsize != -1L,
                     error_type_t::ValidationError,
                     "Error sizing input file! Given path: %s",
                     file.c_str());
  std::vector<char> buf(bufsize + 1);
  rewind(fp.get());

  mps_parser_expects(
    fread(buf.data(), sizeof(char), bufsize, fp.get()) == static_cast<size_t>(bufsize),
    error_type_t::ValidationError,
    "Error reading input file! Given path: %s",
    file.c_str());
  buf[bufsize] = '\0';

  return buf;
}

}  // namespace cuopt::mathematical_optimization::io
