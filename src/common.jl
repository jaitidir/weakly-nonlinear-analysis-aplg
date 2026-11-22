# Shared by all the notebooks: packages, folders, loading/saving data, figure style, legends and videos.
using LinearAlgebra, Statistics, Printf, Random, JLD2, FFTW, Plots, LaTeXStrings, Base64, StableRNGs

# The one source of random numbers of the project: a local generator from an integer seed, passed to rand(rng, ...).
# StableRNG gives the same numbers on every Julia version and machine, and the global generator is never touched.
seeded_rng(seed::Integer) = StableRNG(seed)

const ROOT = dirname(@__DIR__)
const DATA = joinpath(ROOT, "data")        # data/<notebook>/<file>.jld2
const IMAGES = joinpath(ROOT, "images")    # every figure, as pdf
const VIDEOS = joinpath(ROOT, "videos")    # every video, as mp4

# Include a file of src once (the notebook files share the numerical ones)
const LOADED = Set{String}()
function load_src(name)
    name in LOADED && return
    push!(LOADED, name)
    Base.include(Main, joinpath(@__DIR__, name))
end

# ------------------------------------------------------------------------------------------------------------------
# Data: each result is a named tuple saved as data/<file> (one key per field)
# ------------------------------------------------------------------------------------------------------------------

# compute == true: run f(), save its result in data/<file> and return it; false: load data/<file>
function getdata(f, file::String, compute::Bool)
    compute || return loaddata(file)
    t = @elapsed d = f()
    savedata(file, d)
    @printf("computed and saved data/%s (%.0f s)\n", file, t)
    return d
end

savedata(file, d::NamedTuple) = (path = joinpath(DATA, file); mkpath(dirname(path));
                                 jldsave(path; (k => v for (k, v) in pairs(d))...))

# The saved file as a named tuple again (field order may differ from the saved one; fields are read by name)
function loaddata(file)
    path = joinpath(DATA, file)
    isfile(path) || error("data/$file not found: set its compute flag to true to compute it")
    return (; (Symbol(k) => v for (k, v) in JLD2.load(path))...)
end

# ------------------------------------------------------------------------------------------------------------------
# Figures: GR, Computer Modern text, pdf in images/
# ------------------------------------------------------------------------------------------------------------------

const FONT = "Computer Modern"
gr()
default(fontfamily=FONT, grid=true, gridalpha=1.0, gridcolor=:lightgray, background_color=:white)

# Text in Computer Modern (annotations do not take the default font)
txt(s, size, args...) = Plots.text(s, size, args..., FONT)

# Pixels (at 100 dpi) of a length in mm
px(l) = l.value / 0.254                  # 1 px = 0.254 mm at 100 dpi

# Save as images/<name>.pdf and return the figure. While saving, the pdf is made to look like the png shown in the notebook
function savepdf(fig, name)
    mkpath(IMAGES)
    Plots.prepare_output(fig)                    # lay the figure out, so that axis limits and plot areas are known
    # original stroke widths and data of every series, put back after saving
    saved = [(s, s[:markerstrokewidth], s[:x], s[:y]) for sp in fig.subplots for s in sp.series_list]
    k = 500 / minimum(fig[:size])                # marker strokes scaled so that the pdf matches the png (GR quirk)
    for sp in fig.subplots
        xl = Plots.axis_limits(sp, :x); yl = Plots.axis_limits(sp, :y); pa = Plots.plotarea(sp)
        sx = px(Plots.width(pa)) / (xl[2] - xl[1]); sy = px(Plots.height(pa)) / (yl[2] - yl[1])     # px per unit
        for s in sp.series_list
            s[:markerstrokewidth] = s[:markerstrokewidth] .* k
            if s[:markershape] != :none          # markers on an axis limit: moved 1e-9 of the range inside
                s[:x] = inside.(s[:x], xl...); s[:y] = inside.(s[:y], yl...)
            end
            # dashed/dotted curves on linear axes: points closer than 0.5 px dropped
            if s[:seriestype] == :path && s[:linestyle] != :solid && length(s[:x]) > 2 &&
               sp[:xaxis][:scale] == sp[:yaxis][:scale] == :identity
                s[:x], s[:y] = thin_points(s[:x], s[:y], sx, sy)
            end
        end
    end
    savefig(fig, joinpath(IMAGES, name * ".pdf"))
    for (s, msw, x, y) in saved                  # the figure object gets its original data back
        s[:markerstrokewidth] = msw; s[:x] = x; s[:y] = y
    end
    return fig
end

# v moved 1e-9 of the axis (lo, hi) inside if it lies on a limit
inside(v, lo, hi) = (e = 1e-9 * (hi - lo); lo - e <= v < lo + e ? lo + e : hi - e < v <= hi + e ? hi - e : v)

# Points of a curve at least d px apart (sx, sy: px per unit), keeping its ends and its NaN breaks
function thin_points(x, y, sx, sy; d=0.5)
    keep = [1]
    for i in 2:length(x)
        # distance in px to the last kept point (NaN at and right after a break, so both are kept)
        dist = hypot(sx * (x[i] - x[keep[end]]), sy * (y[i] - y[keep[end]]))
        i == length(x) && dist < d && length(keep) > 1 && pop!(keep)     # (the last point replaces a too close one)
        (dist >= d || isnan(dist) || i == length(x)) && push!(keep, i)
    end
    return x[keep], y[keep]
end

# Arrow from (x0, y0) to (x1, y1) with a filled head of length hl and half-width hw in px, in a panel of W × H px with
# axis limits xl, yl
function arrow!(p, x0, y0, x1, y1, xl, yl, (W, H); color=:black, lw=1.7, hl=12, hw=4)
    sx = (xl[2] - xl[1]) / W; sy = (yl[2] - yl[1]) / H                 # data units per px
    ux, uy = (x1 - x0) / sx, (y1 - y0) / sy; L = hypot(ux, uy); ux /= L; uy /= L     # unit direction in px
    bx, by = x1 - hl * ux * sx, y1 - hl * uy * sy                       # base of the head; its corners ±hw px across
    plot!(p, [x0, bx], [y0, by]; color=color, lw=lw, label="")
    plot!(p, Shape([x1, bx - hw * uy * sx, bx + hw * uy * sx], [y1, by + hw * ux * sy, by - hw * ux * sy]); color=color,
          linecolor=color, label="")
end

# ------------------------------------------------------------------------------------------------------------------
# Legends: one framed legend above the panels of a figure, with the entries in a grid of columns
# ------------------------------------------------------------------------------------------------------------------

const LEGEND_KEYS = (:seriestype, :linestyle, :linecolor, :linewidth, :markershape, :markercolor, :markerstrokecolor,
                     :markersize, :markerstrokewidth)

# Legend entries (label, style) of the labelled series of some panels, first occurrence of each label
function legend_entries(panels...)
    seen = Set{String}(); entries = Tuple{Any,Dict{Symbol,Any}}[]
    for p in panels, sp in p.subplots, s in sp.series_list
        lab = s[:label]
        (isempty(string(lab)) || string(lab) in seen) && continue
        push!(seen, string(lab))
        push!(entries, (lab, Dict{Symbol,Any}(k => s[k] for k in LEGEND_KEYS)))
    end
    return entries
end

# Legend entry with the given style (line: color, linestyle, linewidth; marker: marker, markercolor, ...)
function legend_entry(lab; color=:black, linestyle=:solid, linewidth=1, marker=:none, markercolor=:black,
                      markerstrokecolor=:black, markersize=4, markerstrokewidth=1)
    return (lab, Dict{Symbol,Any}(:seriestype => marker == :none ? :path : :scatter, :linestyle => linestyle,
                                  :linecolor => color, :linewidth => linewidth, :markershape => marker,
                                  :markercolor => markercolor, :markerstrokecolor => markerstrokecolor,
                                  :markersize => markersize, :markerstrokewidth => markerstrokewidth))
end

# Visible length of a label in characters of plain text (the unit the legend widths are calibrated in). A LaTeX label
# (with $) is measured as GR draws it, relative to a plain reference text: counting its characters without the markup
# overestimated it (e.g. the subscript in φ_a), pushing the next legend column to the right
function visible_length(lab)
    s = string(lab)
    occursin('$', s) || return length(s)
    GR = Plots.GR
    GR.settextfontprec(Plots.gr_font_family[lowercase(FONT)], GR.TEXT_PRECISION_OUTLINE); GR.setcharheight(0.02)
    ref = "numerical continuation"
    return length(ref) * Plots.gr_text_size(lab isa LaTeXString ? lab : latexstring(s))[1] / Plots.gr_text_size(ref)[1]
end

# Draw the entries into subplot `leg` (axes 0..1 over the whole figure width figwidth, in px): a centred grid filled row
# by row, each column as wide as its longest label, in a thin frame. sample: length of the line samples (fraction of
# the width), pad: space between the frame and the entries
function draw_legend!(leg, entries, figwidth; columns=3, fontsize=18, sample=0.035, pad=0.015, cw_scale=1.0,
                      rowspan=0.76, gap=0.006)
    # mean character width as a fraction of the figure width (≈ 0.52 em; pt -> px at 100 dpi)
    cw = cw_scale * 0.52 * fontsize * (100 / 72) / figwidth
    spacing = 0.04                               # space between columns (gap: from a sample to its label)
    ncol = min(columns, length(entries)); nrow = cld(length(entries), ncol)
    widths = [sample + gap + cw * visible_length(lab) for (lab, _) in entries]
    colw = [maximum(widths[c:ncol:end]) for c in 1:ncol]     # entries c, c+ncol, ... are in column c
    total = sum(colw) + spacing * (ncol - 1)
    left = (1 - total) / 2
    for (i, (lab, a)) in enumerate(entries)
        r, c = divrem(i - 1, ncol)               # 0-based row and column, filled row by row
        x = left + sum(colw[1:c]; init=0.0) + spacing * c
        y = 0.5 + rowspan / 2 - (r + 0.5) * rowspan / nrow
        if a[:seriestype] == :scatter || a[:markershape] ∉ (:none, nothing)
            scatter!(leg, [x + sample / 2], [y]; label="", markershape=a[:markershape], markercolor=a[:markercolor],
                     markerstrokecolor=a[:markerstrokecolor], markersize=a[:markersize], markerstrokewidth=a[:markerstrokewidth])
        else
            plot!(leg, [x, x + sample], [y, y]; label="", linestyle=a[:linestyle], linecolor=a[:linecolor],
                  linewidth=a[:linewidth])
        end
        annotate!(leg, x + sample + gap, y, txt(lab, fontsize, :black, :left))
    end
    x0 = left - pad; x1 = left + total + pad     # frame: rows band plus 0.07 above and below
    plot!(leg, [x0, x1, x1, x0, x0], [0.5 - rowspan/2 - 0.07, 0.5 - rowspan/2 - 0.07, 0.5 + rowspan/2 + 0.07,
                                      0.5 + rowspan/2 + 0.07, 0.5 - rowspan/2 - 0.07]; label="", color=:black, linewidth=1)
    return leg
end

function with_legend(body, entries; size=(1700, 500), legend_height=0.16, columns=3, fontsize=18, sample=0.035,
                     pad=0.015, gap=0.006, cw_scale=1.0, margin=10Plots.mm, bottom_margin=(margin === nothing ? nothing : margin + 5Plots.mm), legend_top=2Plots.mm,
                     legend_bottom=0Plots.mm, kw...)
    blank = plot(framestyle=:none, grid=false, legend=false, ticks=false)
    figsize = (size[1], round(Int, size[2] / (1 - legend_height)))     # taller, so the panels keep the height size[2]
    fig = plot(blank, body; layout=grid(2, 1, heights=[legend_height, 1 - legend_height]), size=figsize, kw...)
    if margin !== nothing                     # (nothing: keep the margins of the panels)
        for sp in fig.subplots[2:end]
            plot!(sp; margin=margin, bottom_margin=bottom_margin)
        end
    end
    plot!(fig[1]; top_margin=legend_top, bottom_margin=legend_bottom)
    Plots.prepare_output(fig)
    pa = Plots.plotarea(fig[1]); W = figsize[1] * 0.254      # figure width in mm
    k = length(fig.subplots) + 1
    # inset over 99% of the figure width (from 0.5% of it), in the coordinates of the plot area of the top subplot
    plot!(fig; inset=(1, bbox((0.005W - Plots.left(pa).value) / Plots.width(pa).value, 0.0, 0.99W / Plots.width(pa).value, 1.0)),
          subplot=k, xlims=(0, 1), ylims=(0, 1), framestyle=:none, legend=false, grid=false, ticks=false,
          background_color_inside=nothing)
    draw_legend!(fig[k], entries, 0.99figsize[1]; columns=columns, fontsize=fontsize, sample=sample, pad=pad, gap=gap, cw_scale=cw_scale)
    return fig
end

# ------------------------------------------------------------------------------------------------------------------
# Videos: frames at 200 dpi, saved in videos/
# ------------------------------------------------------------------------------------------------------------------

const VIDEO_DPI = 200                    # frames drawn at 2x the pixels of their size
video_path(name) = (mkpath(VIDEOS); joinpath(VIDEOS, name))

function save_video(anim, name; fps=25)
    fname = video_path(name)
    frames = joinpath(anim.dir, "%06d.png")
    # H.264, high quality (crf 16); the pad filter rounds the frame size up to even numbers, as yuv420p requires
    Plots.FFMPEG.exe(`-loglevel error -y -framerate $fps -i $frames -c:v libx264 -preset slow -crf 16 -tune animation -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -movflags +faststart $fname`)
    @info "saved videos/$name ($(length(anim.frames)) frames, $(round(length(anim.frames)/fps; digits=1)) s)"
    return fname
end

# Show a saved video in the notebook
show_video(name; width=900) = HTML("""<video controls loop style="width:100%; max-width:$(width)px; height:auto" src="data:video/mp4;base64,$(base64encode(read(video_path(name))))"></video>""")
