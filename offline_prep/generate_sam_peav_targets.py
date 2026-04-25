"""
Loads M1's .pt files, runs SAM/PE-AV, saves target .pt files.

Things to do:
1. Take bboxes from M1 -> crop the image using bboxes
 -> pass to peav -> will be our vanilla PEAV embeddings. Our baseline.
2. Take bboxes -> crop -> apply SAM segmentation -> get mask -> blackout background -> pass to peav -> will be our sam_no_context embeddings.
3. Take bboxes -> crop -> apply SAM segmentation -> get mask -> blackout background -> pass to peav -> all pass full image to peav -> average the embeddings -> sam_with_context_equal
4. same as 3 but averaging with 80 20 ratio -> sam_with_context_80_20 embedding
"""