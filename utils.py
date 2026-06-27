def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    """
    Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.
    """
    if not isinstance(pipeline_output, dict):
        raise TypeError(
            f"Expected pipeline_output to be a dict, got {type(pipeline_output).__name__}"
        )
    if "chunks" not in pipeline_output:
        raise KeyError(
            "pipeline_output is missing the 'chunks' key. "
            "Ensure the pipeline was called with return_timestamps='word'."
        )

    adjusted_chunks = pipeline_output["chunks"].copy()

    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]

        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]

        if any(v is None for v in (current_start, current_end, next_start, next_end)):
            continue

        pause_duration = next_start - current_end

        if pause_duration > 0:
            if pause_duration > split_threshold:
                distribute = split_threshold / 2
            else:
                distribute = pause_duration / 2

            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)
            adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)

    pipeline_output["chunks"] = adjusted_chunks

    return pipeline_output
