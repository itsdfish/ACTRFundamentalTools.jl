function plot_chains(chain; options...)
    p1 = plot(chain, xaxis=font(5), yaxis=font(5), seriestype=(:traceplot),
      grid=false, size=(275,125), titlefont=font(5); options...)
    p2 = plot(chain, xaxis=font(5), yaxis=font(5), seriestype=(:autocorplot),
      grid=false, size=(275,125), titlefont=font(5); options...)
    p3 = plot(chain, xaxis=font(5), yaxis=font(5), seriestype=(:mixeddensity),
      grid=false, size=(275,125), titlefont=font(5); options...)
    p = plot(p1, p2, p3, layout=(3,1), size=(300,300); options...)
    return p
end
