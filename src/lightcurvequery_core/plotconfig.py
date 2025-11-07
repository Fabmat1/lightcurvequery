# In plotconfig.py
import yaml
from dataclasses import dataclass, field
from typing import Optional
from .title_manager import TitleTemplate

@dataclass
class PlotConfig:
    """Configuration for plot styling."""
    # ======= Fonts =======
    title_fontsize: int = 12
    label_fontsize: int = 12
    legend_fontsize: int = 8
    tick_fontsize: int = 10
    
    # ======= Figure =======
    figsize: Optional[tuple[float, float]] = None
    dpi: int = 100
    facecolor: str = "white"
    
    # ======= Grid =======
    grid: bool = True
    grid_linestyle: str = "--"
    grid_color: str = "darkgrey"
    grid_alpha: float = 1.0
    
    # ======= Markers (for photometry plots) =======
    marker_size: float = 3
    marker_alpha: float = 1.0
    marker_style: str = '.'
    errorbar_capsize: float = 3
    errorbar_width: float = 0.5
    
    # ======= Lines (for periodogram & fits) =======
    fit_line_width: float = 2
    fit_line_color: str = "mediumblue"
    fit_line_alpha: float = 1.0
    
    periodogram_line_width: float = 1.5
    periodogram_line_color: str = "#6D23B6"
    periodogram_line_alpha: float = 1.0
    
    peak_line_style: str = "--"
    peak_line_color: str = "red"
    peak_line_width: float = 1.5
    peak_line_alpha: float = 0.8
    
    hpd_line_style: str = "--"
    hpd_line_color: str = "black"
    hpd_line_width: float = 1.5
    hpd_line_alpha: float = 0.8
    
    # ======= Colors =======
    band_colors: list[str] = field(default_factory=lambda: [
        "darkred", "navy", "darkgreen", "violet", "magenta",
        "gold", "salmon", "brown", "black", "lime", "red",
    ])
    
    # ======= Overrides =======
    telescope_overrides: dict = field(default_factory=dict)
    
    # ======= Titles & Labels =======
    main_title: Optional[str] = None
    ylabel: str = "Normalized flux"
    xlabel: str = "Phase"
    show_title: bool = True
    title_template: Optional[TitleTemplate] = None
    show_legend: bool = True
    show_ylabel: bool = True
    show_xlabel: bool = True
    
    # ======= Annotations =======
    annotation_fontsize: int = 8
    annotation_bgcolor: str = "white"
    annotation_alpha: float = 0.6
    
    # ======= Layout =======
    hspace: float = 0.0
    tight_layout: bool = False  # Changed default to False
    constrained_layout: bool = True  # NEW: Use constrained_layout instead
    xlabel_pad: float = 0.01  # Distance from bottom (0-1)
    ylabel_pad: float = 0.02  # Distance from left (0-1)
    title_pad: float = 0.98   # Distance from bottom for title (0-1)

    # NEW: Fine-tune subplot positions
    left_margin: float = 0.1
    right_margin: float = 0.95
    top_margin: float = 0.95
    bottom_margin: float = 0.08
    
    def get_title_manager(self) -> TitleTemplate:
        """Get the title template manager."""
        if self.title_template is None:
            return TitleTemplate(show_titles=self.show_title)
        return self.title_template
    
    @classmethod
    def from_file(cls, path: str) -> 'PlotConfig':
        """Load config from YAML or JSON file."""
        import json
        if path.endswith('.json'):
            with open(path) as f:
                data = json.load(f)
        else:
            with open(path) as f:
                data = yaml.safe_load(f)
        
        # Remove title_template from data if present (can't deserialize it from file)
        data.pop('title_template', None)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def merge(self, other: 'PlotConfig') -> 'PlotConfig':
        """Merge another config, with other taking precedence."""
        from copy import deepcopy
        merged = deepcopy(self)
        for key, value in vars(other).items():
            if value is not None or not hasattr(merged, key):
                setattr(merged, key, value)
        return merged


STYLE_PRESETS = {
    'paper_single_column': PlotConfig(
        figsize=(3.46, 4.5),
        title_fontsize=9,
        label_fontsize=8,
        legend_fontsize=0,  
        tick_fontsize=7,
        annotation_fontsize=0,
        marker_size=2,
        errorbar_capsize=2,
        errorbar_width=0.8,
        dpi=300,
        grid=False,
        show_title=False,
        show_legend=False,
        show_ylabel=True,
        show_xlabel=True,
        periodogram_line_width=1.0,
        fit_line_width=1.0,
        constrained_layout=True,
        tight_layout=False,
        left_margin=0.15,
        right_margin=0.95,
        top_margin=0.95,
        bottom_margin=0.12,
    ),
    'paper_double_column': PlotConfig(
        figsize=(7.09, 4.5),
        title_fontsize=10,
        label_fontsize=9,
        legend_fontsize=7,
        tick_fontsize=8,
        annotation_fontsize=0,
        marker_size=2.5,
        errorbar_capsize=2.5,
        errorbar_width=0.8,
        dpi=300,
        grid=False,
        show_title=False,
        show_legend=False,
        show_ylabel=True,
        show_xlabel=True,
        periodogram_line_width=1.0,
        fit_line_width=1.0,
        constrained_layout=True,
        tight_layout=False,
        left_margin=0.1,
        right_margin=0.95,
        top_margin=0.95,
        bottom_margin=0.1,
    ),
    'presentation': PlotConfig(
        figsize=(10, 5.625),
        title_fontsize=18,
        label_fontsize=16,
        legend_fontsize=10,
        tick_fontsize=14,
        annotation_fontsize=12,
        marker_size=5,
        errorbar_capsize=4,
        errorbar_width=1.0,
        dpi=100,
        grid=True,
        show_title=True,
        show_legend=True,
        show_ylabel=True,
        show_xlabel=True,
        periodogram_line_width=2.0,
        fit_line_width=2.5,
        constrained_layout=True,
        tight_layout=False,
    ),
    'minimal': PlotConfig(
        figsize=None,
        title_fontsize=10,
        label_fontsize=9,
        legend_fontsize=0,
        tick_fontsize=8,
        annotation_fontsize=7,
        marker_size=2,
        dpi=100,
        grid=False,
        show_title=False,
        show_legend=False,
        show_ylabel=False,
        show_xlabel=False,
        periodogram_line_width=1.5,
        fit_line_width=2.0,
        constrained_layout=False,
        tight_layout=False,
        left_margin=0.08,
        right_margin=0.98,
        top_margin=0.98,
        bottom_margin=0.05,
    ),
}