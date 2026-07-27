/**
 * BridgeGuardian AI — Client-Side Image Compression Utility
 * Resizes and compresses high-resolution camera images in the browser
 * to prevent massive multipart payload timeouts during backend upload.
 */

export async function compressImage(
  file: File,
  maxWidth = 1920,
  maxHeight = 1920,
  quality = 0.85
): Promise<File> {
  // If file is not an image, return unchanged
  if (!file.type.startsWith('image/')) return file

  // For small images (< 500 KB), return directly without re-encoding
  if (file.size < 500 * 1024) return file

  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const img = document.createElement('img')
      img.onload = () => {
        let width = img.width
        let height = img.height

        // Calculate aspect ratio scaling
        if (width > maxWidth || height > maxHeight) {
          if (width > height) {
            height = Math.round((height * maxWidth) / width)
            width = maxWidth
          } else {
            width = Math.round((width * maxHeight) / height)
            height = maxHeight
          }
        }

        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height
        const ctx = canvas.getContext('2d')
        if (!ctx) return resolve(file)

        ctx.drawImage(img, 0, 0, width, height)

        canvas.toBlob(
          (blob) => {
            if (!blob) return resolve(file)
            const compressedFile = new File([blob], file.name.replace(/\.[^/.]+$/, '.jpg'), {
              type: 'image/jpeg',
              lastModified: Date.now(),
            })
            resolve(compressedFile)
          },
          'image/jpeg',
          quality
        )
      }

      img.onerror = () => resolve(file)
      img.src = e.target?.result as string
    }

    reader.onerror = () => resolve(file)
    reader.readAsDataURL(file)
  })
}

export async function compressImageBatch(
  files: File[],
  maxWidth = 1920,
  maxHeight = 1920,
  quality = 0.85
): Promise<File[]> {
  return Promise.all(files.map((file) => compressImage(file, maxWidth, maxHeight, quality)))
}
