ninja -j16 ckProfiler
# stdbuf -oL -eL ../run_fwd.sh &> ../OUT_CK_FWD
stdbuf -oL -eL ../run_tile_wrw.sh &> ../OUT_CK_TILE_WRW
# stdbuf -oL -eL ../run_tile_fwd.sh &> ../OUT_CK_TILE_FWD
# stdbuf -oL -eL ../run_wrw.sh &> ../OUT_CK_WRW
# stdbuf -oL -eL ../run_bwd.sh &> ../OUT_CK_BWD
# stdbuf -oL -eL ../run_tile_bwd.sh &> ../OUT_CK_TILE_BWD