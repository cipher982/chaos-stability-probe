local figures = {
  ["Figure 1"] = {
    src = "../figures/signature_rescue_panel.png",
    alt = "Figure 1. Signature rescue panel",
  },
  ["Figure 2"] = {
    src = "../figures/branch_card_hero.png",
    alt = "Figure 2. Branch Card hero screenshot",
  },
}

function Para(el)
  local text = pandoc.utils.stringify(el)

  for label, fig in pairs(figures) do
    if text:match("^" .. label) then
      local img = pandoc.Image({ pandoc.Str(fig.alt) }, fig.src)
      img.attr = pandoc.Attr("", { "paper-figure" }, {})
      return {
        pandoc.Plain({ img }),
        el,
      }
    end
  end
end
