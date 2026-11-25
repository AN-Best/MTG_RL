"""Merge Figures 1-6 into a single shareable PDF: figures/all_figures.pdf."""

import os

PDF_FILES = [
    "figures/fig1_control_loop.pdf",
    "figures/fig2_walking.pdf",
    "figures/fig3_running.pdf",
    "figures/fig4_elderly_jump.pdf",
    "figures/fig5_5xsts.pdf",
    "figures/fig6_walk_slip.pdf",
]

PNG_FILES = [
    "figures/fig1_control_loop.png",
    "figures/fig2_walking.png",
    "figures/fig3_running.png",
    "figures/fig4_elderly_jump.png",
    "figures/fig5_5xsts.png",
    "figures/fig6_walk_slip.png",
]

OUT_PDF = "figures/all_figures.pdf"


def merge_vector_pdfs():
    for mod_name in ("pypdf", "PyPDF2"):
        try:
            mod = __import__(mod_name)
            writer = mod.PdfWriter()
            for p in PDF_FILES:
                writer.append(p)
            with open(OUT_PDF, "wb") as f:
                writer.write(f)
            print(f"[DONE] Merged 6 vector PDFs into {OUT_PDF} via {mod_name}")
            return True
        except Exception:
            pass
    try:
        import fitz  # PyMuPDF
        out = fitz.open()
        for p in PDF_FILES:
            with fitz.open(p) as doc:
                out.insert_pdf(doc)
        out.save(OUT_PDF)
        out.close()
        print(f"[DONE] Merged 6 vector PDFs into {OUT_PDF} via PyMuPDF")
        return True
    except Exception:
        return False


def merge_via_pdfpages():
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(OUT_PDF) as pdf:
        for png in PNG_FILES:
            img = mpimg.imread(png)
            h, w = img.shape[:2]
            fig = plt.figure(figsize=(w / 300.0, h / 300.0), dpi=300)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
    print(f"[DONE] Compiled 6 300-DPI figures into {OUT_PDF} via matplotlib PdfPages")


def main():
    os.makedirs("figures", exist_ok=True)
    if not merge_vector_pdfs():
        merge_via_pdfpages()
    size_mb = os.path.getsize(OUT_PDF) / (1024 * 1024)
    print(f"  Output size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
