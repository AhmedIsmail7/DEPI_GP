from modules.transcribe import compute_chunk_boundaries


class TestComputeChunkBoundaries:
    def test_exact_fit(self):
        # 60s audio, 30s chunks, no overlap -> 2 clean chunks
        result = compute_chunk_boundaries(total_ms=60_000, chunk_ms=30_000, overlap_ms=0)
        assert result == [(0, 30_000), (30_000, 60_000)]

    def test_with_overlap_matches_real_config(self):
        # Matches CHUNK_DURATION_MS=30000, OVERLAP_MS=5000 from config.py
        result = compute_chunk_boundaries(total_ms=100_000, chunk_ms=30_000, overlap_ms=5_000)
        step = 30_000 - 5_000  # 25000
        expected_starts = list(range(0, 100_000, step))
        assert [s for s, _ in result] == expected_starts
        # every chunk is exactly chunk_ms wide
        assert all(end - start == 30_000 for start, end in result)

    def test_consecutive_chunks_actually_overlap(self):
        result = compute_chunk_boundaries(total_ms=100_000, chunk_ms=30_000, overlap_ms=5_000)
        for i in range(len(result) - 1):
            _, end_current = result[i]
            start_next, _ = result[i + 1]
            assert start_next < end_current  # confirms real overlap, not a gap

    def test_short_audio_produces_one_chunk(self):
        result = compute_chunk_boundaries(total_ms=10_000, chunk_ms=30_000, overlap_ms=5_000)
        assert len(result) == 1
        assert result[0][0] == 0

    def test_zero_duration(self):
        result = compute_chunk_boundaries(total_ms=0, chunk_ms=30_000, overlap_ms=5_000)
        assert result == []