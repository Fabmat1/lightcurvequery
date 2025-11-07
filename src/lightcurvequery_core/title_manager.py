"""Title management with template-based customization."""
from typing import Optional


class TitleTemplate:
    """Manages plot titles with template variable substitution."""
    
    DEFAULT_PHOTOMETRY = "Lightcurve for Gaia DR3 {gaia_id}"
    DEFAULT_PERIODOGRAM = "Periodograms for Gaia DR3 {gaia_id}"
    DEFAULT_RV = "Radial Velocity for Gaia DR3 {gaia_id}"
    
    AVAILABLE_VARIABLES = {
        'gaia_id': 'Gaia DR3 source identifier',
        'alias': 'Custom target alias (if provided)',
        'display_name': 'Alias if available, otherwise gaia_id',
    }
    
    def __init__(
        self,
        photometry_template: Optional[str] = None,
        periodogram_template: Optional[str] = None,
        rv_template: Optional[str] = None,
        show_titles: bool = True,
    ):
        """
        Initialize title templates.
        
        Args:
            photometry_template: Template for photometry plots
            periodogram_template: Template for periodogram plots
            rv_template: Template for RV plots
            show_titles: Whether to show titles at all
        """
        self.photometry_template = photometry_template or self.DEFAULT_PHOTOMETRY
        self.periodogram_template = periodogram_template or self.DEFAULT_PERIODOGRAM
        self.rv_template = rv_template or self.DEFAULT_RV
        self.show_titles = show_titles
    
    def get_photometry_title(self, star) -> Optional[str]:
        """Get photometry plot title."""
        if not self.show_titles:
            return None
        return self._format_template(self.photometry_template, star)
    
    def get_periodogram_title(self, star) -> Optional[str]:
        """Get periodogram plot title."""
        if not self.show_titles:
            return None
        return self._format_template(self.periodogram_template, star)
    
    def get_rv_title(self, star) -> Optional[str]:
        """Get RV plot title."""
        if not self.show_titles:
            return None
        return self._format_template(self.rv_template, star)
    
    def _format_template(self, template: str, star) -> str:
        """Format a template with star attributes."""
        return template.format(
            gaia_id=star.gaia_id,
            alias=star.alias or "",
            display_name=star.get_display_name(),
        )
    
    @classmethod
    def from_preset(cls, preset: str) -> 'TitleTemplate':
        """Create TitleTemplate from a named preset."""
        presets = {
            'default': cls(),
            'paper': cls(show_titles=False),
            'minimal': cls(
                photometry_template="{display_name}",
                periodogram_template="{display_name}",
                rv_template="{display_name}",
            ),
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        return presets[preset]
    
    def __repr__(self) -> str:
        return (f"TitleTemplate(phot={self.photometry_template!r}, "
                f"pgram={self.periodogram_template!r}, rv={self.rv_template!r})")