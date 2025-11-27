#!/usr/bin/env python3
"""
Service pour générer des rapports PDF à partir de l'analyse des défauts.
"""

import logging
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor, red, orange, yellow, green
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, PageBreak, KeepTogether)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class PDFReportGenerator:
    """
    Génère des rapports PDF à partir des données d'analyse de défauts.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Couleurs par sévérité (défini uniquement si ReportLab est disponible)
        if REPORTLAB_AVAILABLE:
            self.severity_colors = {
                "critique": red,
                "elevee": orange,
                "moderee": yellow,
                "faible": green
            }
        else:
            self.severity_colors = {}

        if not REPORTLAB_AVAILABLE:
            self.logger.warning("ReportLab n'est pas installé. L'export PDF ne sera pas disponible.")

    def is_available(self) -> bool:
        """Vérifie si l'export PDF est disponible."""
        return REPORTLAB_AVAILABLE

    def generate_report(
        self,
        report_data: Dict,
        output_path: str,
        include_defect_details: bool = True,
        max_defects_in_pdf: int = 100
    ) -> bool:
        """
        Génère un rapport PDF à partir des données d'analyse.

        Args:
            report_data: Dictionnaire contenant les résultats de l'analyse
            output_path: Chemin du fichier PDF de sortie
            include_defect_details: Inclure les détails de chaque défaut
            max_defects_in_pdf: Nombre maximum de défauts à inclure dans le PDF

        Returns:
            True si le PDF a été généré avec succès, False sinon
        """
        if not REPORTLAB_AVAILABLE:
            self.logger.error("ReportLab n'est pas installé")
            return False

        try:
            self.logger.info(f"Génération du rapport PDF: {output_path}")

            # Créer le document PDF
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )

            # Container pour les éléments du PDF
            story = []

            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=HexColor('#2E4053'),
                spaceAfter=30,
                alignment=TA_CENTER
            )

            # Ajouter le contenu
            self._add_title(story, title_style)
            self._add_metadata(story, report_data.get('metadata', {}), styles)
            self._add_statistics(story, report_data.get('statistics', {}), styles)

            if include_defect_details:
                defects = report_data.get('defects', [])
                # Limiter le nombre de défauts pour éviter des PDF trop lourds
                defects_to_include = defects[:max_defects_in_pdf]

                self._add_defects_summary(story, defects_to_include, styles)

                if len(defects) > max_defects_in_pdf:
                    warning_text = f"⚠️ Note: Seuls les {max_defects_in_pdf} premiers défauts sont affichés dans ce rapport. " \
                                   f"Total de {len(defects)} défauts détectés."
                    story.append(Paragraph(warning_text, styles['Normal']))
                    story.append(Spacer(1, 0.2 * inch))

            # Générer le PDF
            doc.build(story)

            self.logger.info(f"Rapport PDF généré avec succès: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du PDF: {e}", exc_info=True)
            return False

    def _add_title(self, story, title_style):
        """Ajoute le titre du rapport."""
        title = Paragraph("Rapport d'Analyse des Défauts", title_style)
        story.append(title)
        story.append(Spacer(1, 0.3 * inch))

    def _add_metadata(self, story, metadata: Dict, styles):
        """Ajoute les métadonnées du rapport."""
        story.append(Paragraph("<b>Informations Générales</b>", styles['Heading2']))

        data = [
            ["Fichier NDE:", metadata.get('nde_file', 'N/A')],
            ["Date d'analyse:", metadata.get('analysis_date', 'N/A')],
            ["Classe A:", metadata.get('class_A', 'N/A')],
            ["Classe B:", metadata.get('class_B', 'N/A')],
            ["Nombre d'endviews:", str(metadata.get('num_endviews', 'N/A'))]
        ]

        table = Table(data, colWidths=[2 * inch, 4 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))

        story.append(table)
        story.append(Spacer(1, 0.3 * inch))

    def _add_statistics(self, story, statistics: Dict, styles):
        """Ajoute les statistiques globales."""
        story.append(Paragraph("<b>Statistiques Globales</b>", styles['Heading2']))

        total_defects = statistics.get('total_defects', 0)
        integrity_score = statistics.get('integrity_score', 0.0)
        status = statistics.get('status', 'unknown')
        recommendation = statistics.get('recommendation', 'N/A')

        # Score d'intégrité avec couleur
        integrity_percent = int(integrity_score * 100)
        if integrity_percent >= 80:
            integrity_color = 'green'
        elif integrity_percent >= 60:
            integrity_color = 'orange'
        else:
            integrity_color = 'red'

        summary_text = f"""
        <b>Total de défauts détectés:</b> {total_defects}<br/>
        <b>Score d'intégrité:</b> <font color='{integrity_color}'>{integrity_percent}%</font><br/>
        <b>Statut global:</b> {status.upper()}<br/>
        <b>Recommandation:</b> {recommendation}
        """

        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        # Tableau des défauts par sévérité
        defects_by_severity = statistics.get('defects_by_severity', {})

        if defects_by_severity:
            story.append(Paragraph("<b>Répartition par sévérité:</b>", styles['Normal']))

            severity_data = [["Sévérité", "Nombre"]]
            for severity, count in defects_by_severity.items():
                severity_data.append([severity.capitalize(), str(count)])

            severity_table = Table(severity_data, colWidths=[2 * inch, 1.5 * inch])
            severity_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))

            story.append(severity_table)

        story.append(Spacer(1, 0.3 * inch))

    def _add_defects_summary(self, story, defects: list, styles):
        """Ajoute le tableau récapitulatif des défauts."""
        if not defects:
            story.append(Paragraph("<b>Aucun défaut détecté</b>", styles['Normal']))
            return

        story.append(PageBreak())
        story.append(Paragraph("<b>Liste des Défauts Détectés</b>", styles['Heading2']))
        story.append(Spacer(1, 0.2 * inch))

        # Créer le tableau
        table_data = [["#", "Sévérité", "Type", "Endview", "X", "Description"]]

        for i, defect in enumerate(defects, 1):
            severity = defect.get('severity', 'unknown')
            defect_type = defect.get('type', 'unknown')
            endview_id = defect.get('endview_id', '?')
            x = defect.get('x', '?')
            description = defect.get('description', 'N/A')

            # Tronquer la description si trop longue
            if len(description) > 80:
                description = description[:77] + "..."

            table_data.append([
                str(i),
                severity.capitalize(),
                defect_type,
                str(endview_id),
                str(x),
                description
            ])

        # Créer la table
        defects_table = Table(
            table_data,
            colWidths=[0.4 * inch, 1 * inch, 1.2 * inch, 0.8 * inch, 0.6 * inch, 2.5 * inch]
        )

        # Style du tableau
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]

        # Couleurs par sévérité (lignes alternées avec couleur de sévérité)
        for i, defect in enumerate(defects, 1):
            severity = defect.get('severity', 'unknown')
            color = self.severity_colors.get(severity, colors.lightgrey)
            table_style.append(('BACKGROUND', (1, i), (1, i), color))

        defects_table.setStyle(TableStyle(table_style))

        story.append(defects_table)
        story.append(Spacer(1, 0.3 * inch))

        # Footer
        footer_text = f"<i>Rapport généré le {datetime.now().strftime('%Y-%m-%d à %H:%M:%S')}</i>"
        story.append(Paragraph(footer_text, styles['Normal']))
