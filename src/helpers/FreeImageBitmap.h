#ifndef FREEIMAGEBITMAP_H
#define FREEIMAGEBITMAP_H

#include <FreeImage.h>

class FreeImageBitmap {
private:
    FIBITMAP *bitmap_ = nullptr;

public:
    FreeImageBitmap() = default;

    explicit FreeImageBitmap(FIBITMAP *bmp) : bitmap_(bmp) {
    }

    ~FreeImageBitmap() {
        if (bitmap_) {
            FreeImage_Unload(bitmap_);
        }
    }

    FreeImageBitmap(const FreeImageBitmap &) = delete;
    FreeImageBitmap &operator=(const FreeImageBitmap &) = delete;

    FreeImageBitmap(FreeImageBitmap &&other) noexcept : bitmap_(other.bitmap_) {
        other.bitmap_ = nullptr;
    }

    FreeImageBitmap &operator=(FreeImageBitmap &&other) noexcept {
        if (this != &other) {
            if (bitmap_) {
                FreeImage_Unload(bitmap_);
            }
            bitmap_ = other.bitmap_;
            other.bitmap_ = nullptr;
        }
        return *this;
    }

    FreeImageBitmap &operator=(FIBITMAP *bmp) {
        if (bitmap_ != bmp) {
            if (bitmap_) {
                FreeImage_Unload(bitmap_);
            }
            bitmap_ = bmp;
        }
        return *this;
    }


    FIBITMAP *get() { return bitmap_; }
    [[nodiscard]] FIBITMAP *get() const { return bitmap_; }

    FIBITMAP *release() {
        FIBITMAP *temp = bitmap_;
        bitmap_ = nullptr;
        return temp;
    }

    void reset(FIBITMAP *bmp = nullptr) {
        if (bitmap_) {
            FreeImage_Unload(bitmap_);
        }
        bitmap_ = bmp;
    }

    explicit operator bool() const { return bitmap_ != nullptr; }

    FIBITMAP **address_of() {
        if (bitmap_) {
            FreeImage_Unload(bitmap_);
            bitmap_ = nullptr;
        }
        return &bitmap_;
    }
};

#endif // FREEIMAGEBITMAP_H
