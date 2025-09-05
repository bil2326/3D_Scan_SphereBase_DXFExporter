import streamlit as st
import pandas as pd
import numpy as np
import ezdxf
from io import StringIO
import tempfile
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import logging
import os

# Configuration de logging pour debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SphereCalculator:
    """Classe pour les calculs de positions des sphères"""
    
    def __init__(self, radius: float, base_height: float):
        self.radius = radius
        self.base_height = base_height
        
    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalise un vecteur avec gestion des erreurs"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Impossible de normaliser un vecteur nul")
        return vector / norm
        
    def calculate_base_position(self, center: np.ndarray, normal: np.ndarray, 
                              orientation: str, reference: np.ndarray) -> np.ndarray:
        """Calcule la position de la base d'une sphère"""
        try:
            n = self.normalize_vector(normal)
            
            # Ajustement de direction basé sur la référence
            if np.dot(center - reference, n) < 0:
                n = -n
                
            # Calcul position selon orientation
            if orientation.strip().lower() == 'up':
                return center - (self.radius + self.base_height) * n
            elif orientation.strip().lower() == 'down':
                return center + (self.radius + self.base_height) * n
            else:
                raise ValueError(f"Orientation invalide: {orientation}")
                
        except Exception as e:
            logger.error(f"Erreur calcul position base: {e}")
            raise

class DataValidator:
    """Validation et nettoyage des données d'entrée"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
        """Valide la structure du DataFrame"""
        expected_columns = ['Nom', 'X', 'Y', 'Z', 'nx', 'ny', 'nz', 'Orientation']
        
        if df.empty:
            return False, "DataFrame vide"
            
        if len(df.columns) != len(expected_columns):
            return False, f"Nombre de colonnes incorrect. Attendu: {len(expected_columns)}, trouvé: {len(df.columns)}"
            
        # Vérification des types numériques
        numeric_cols = ['X', 'Y', 'Z', 'nx', 'ny', 'nz']
        for col in numeric_cols:
            try:
                pd.to_numeric(df[col], errors='raise')
            except:
                return False, f"Colonne {col} contient des valeurs non numériques"
                
        # Vérification des orientations
        valid_orientations = ['up', 'down']
        invalid_orientations = df[~df['Orientation'].str.strip().str.lower().isin(valid_orientations)]
        if not invalid_orientations.empty:
            return False, f"Orientations invalides trouvées: {invalid_orientations['Orientation'].tolist()}"
            
        return True, "Validation réussie"

class DXFGenerator:
    """Générateur de fichiers DXF avec gestion robuste"""
    
    def __init__(self, cross_size: float = 50, text_height: float = 200):
        self.cross_size = cross_size
        self.text_height = text_height
        
    def add_cross_and_label(self, msp, x: float, y: float, label: str, color: int = 3):
        """Ajoute une croix et un label au DXF"""
        try:
            # Croix
            half_size = self.cross_size / 2
            msp.add_line(
                (x - half_size, y, 0), 
                (x + half_size, y, 0), 
                dxfattribs={'color': color}
            )
            msp.add_line(
                (x, y - half_size, 0), 
                (x, y + half_size, 0), 
                dxfattribs={'color': color}
            )
            
            # Label
            msp.add_text(
                label,
                dxfattribs={
                    'height': self.text_height,
                    'insert': (x + self.cross_size, y + self.cross_size, 0),
                    'color': color
                }
            )
        except Exception as e:
            logger.error(f"Erreur ajout croix/label: {e}")
            raise
            
    def add_connecting_lines(self, msp, points: List[Tuple[float, float]], color: int = 3):
        """Ajoute des lignes entre les points"""
        try:
            for i in range(len(points) - 1):
                msp.add_line(
                    (points[i][0], points[i][1], 0),
                    (points[i+1][0], points[i+1][1], 0),
                    dxfattribs={'color': color}
                )
        except Exception as e:
            logger.error(f"Erreur ajout lignes: {e}")
            raise

def parse_input_data(data_text: str) -> Optional[pd.DataFrame]:
    """Parse les données d'entrée avec gestion robuste des erreurs"""
    if not data_text.strip():
        return None
        
    try:
        # Détection automatique du séparateur
        df = pd.read_csv(StringIO(data_text), sep=None, engine='python', header=None)
        df.columns = ['Nom', 'X', 'Y', 'Z', 'nx', 'ny', 'nz', 'Orientation']
        
        # Validation
        is_valid, message = DataValidator.validate_dataframe(df)
        if not is_valid:
            st.error(f"Erreur validation données: {message}")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"Erreur lecture données: {str(e)}")
        logger.error(f"Erreur parse données: {e}")
        return None

def create_visualization(relative_positions: np.ndarray, names: List[str], 
                        reference_index: int, inverse_x: bool = False):
    """Crée la visualisation matplotlib"""
    try:
        X_plot = -relative_positions[:, 0] if inverse_x else relative_positions[:, 0]
        Z_plot = relative_positions[:, 2]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Points
        for i, (x, z) in enumerate(zip(X_plot, Z_plot)):
            color = 'green' if i == reference_index else 'red'
            size = 100 if i == reference_index else 80
            ax.scatter(x, z, color=color, s=size, alpha=0.7, edgecolors='black')
            
            # Labels avec offset pour éviter superposition
            offset_x = 0.001 if x >= 0 else -0.001
            offset_z = 0.001
            ax.annotate(names[i], (x, z), xytext=(x + offset_x, z + offset_z), 
                       fontsize=9, ha='left' if x >= 0 else 'right',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Connexions entre points
        if len(X_plot) > 1:
            ax.plot(X_plot, Z_plot, 'b--', alpha=0.5, linewidth=1)
            
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Z (m)', fontsize=12)
        ax.set_title('Positions relatives des bases (Vue X-Z)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Légende
        ax.legend(['Référence', 'Autres points', 'Connexions'], 
                 loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Erreur création visualisation: {e}")
        st.error(f"Erreur lors de la création de la visualisation: {str(e)}")
        return None

# =====================================
# INTERFACE STREAMLIT PRINCIPALE
# =====================================

def main():
    st.set_page_config(
        page_title="Calculateur DXF - Bases de Sphères",
        page_icon="⚙️",
        layout="wide"
    )
    
    st.title("⚙️ Calcul et Export DXF des Bases de Sphères")
    st.markdown("---")
    
    # Sidebar pour paramètres
    with st.sidebar:
        st.header("Paramètres")
        R = st.number_input("Rayon de la sphère (m)", value=0.0725, min_value=0.001, step=0.001, format="%.4f")
        h_base = st.number_input("Hauteur de la base magnétique (m)", value=0.008, min_value=0.001, step=0.001, format="%.4f")
        
        # Paramètres DXF
        st.subheader("Paramètres DXF")
        cross_size = st.number_input("Taille des croix (mm)", value=50, min_value=1)
        text_height = st.number_input("Hauteur du texte (mm)", value=200, min_value=1)
    
    # Zone principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Données d'Entrée")
        st.markdown("**Format:** `Nom X Y Z nx ny nz Orientation` (séparés par tabulations ou espaces)")
        st.markdown("**Orientation:** `up` ou `down`")
        
        data_input = st.text_area(
            "Collez vos données Excel",
            height=250,
            placeholder="Nom\tX\tY\tZ\tnx\tny\tnz\tOrientation\nPoint1\t1.0\t2.0\t3.0\t0\t0\t1\tup\n...",
            help="Copiez-collez directement depuis Excel"
        )
    
    # Traitement des données
    if not data_input.strip():
        st.info("👆 Veuillez saisir vos données pour commencer")
        return
    
    df = parse_input_data(data_input)
    if df is None:
        return
    
    # Affichage des données parsées
    with col2:
        st.subheader("📋 Données Parsées")
        st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    
    # Choix de référence
    st.subheader("🎯 Choix de la Référence")
    noms = df['Nom'].tolist()
    index_ref = st.selectbox(
        "Choisir la référence :",
        options=range(len(noms)),
        format_func=lambda x: f"{noms[x]} (ligne {x+1})"
    )
    
    # Calculs
    try:
        calculator = SphereCalculator(R, h_base)
        C_ref = df.loc[index_ref, ['X', 'Y', 'Z']].values.astype(float)
        
        positions_base = []
        for i, row in df.iterrows():
            C = row[['X', 'Y', 'Z']].values.astype(float)
            n_raw = row[['nx', 'ny', 'nz']].values.astype(float)
            orientation = row['Orientation']
            
            P_base = calculator.calculate_base_position(C, n_raw, orientation, C_ref)
            positions_base.append(P_base)
        
        positions_base = np.array(positions_base)
        relative_positions = positions_base - positions_base[index_ref]
        
    except Exception as e:
        st.error(f"Erreur lors des calculs: {str(e)}")
        return
    
    # Affichage des résultats
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📍 Positions Absolues (m)")
        for i, pos in enumerate(positions_base):
            icon = "🎯" if i == index_ref else "📍"
            st.write(f"{icon} **{df.loc[i,'Nom']}**: X={pos[0]:.4f}, Y={pos[1]:.4f}, Z={pos[2]:.4f}")
    
    with col2:
        st.subheader(f"📐 Positions Relatives à {df.loc[index_ref,'Nom']} (m)")
        for i, pos in enumerate(relative_positions):
            icon = "🎯" if i == index_ref else "📐"
            st.write(f"{icon} **{df.loc[i,'Nom']}**: dX={pos[0]:.4f}, dY={pos[1]:.4f}, dZ={pos[2]:.4f}")
    
    # Visualisation
    st.markdown("---")
    st.subheader("📈 Visualisation")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        inverse_x = st.checkbox("🔄 Inverser l'axe X", value=False)
        
    fig = create_visualization(relative_positions, noms, index_ref, inverse_x)
    if fig:
        st.pyplot(fig)
    
    # Export DXF
    st.markdown("---")
    st.subheader("🎨 Export DXF")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        dxf_file = st.file_uploader("📎 Plan DXF existant", type=["dxf"])
        ref_x_autocad = st.number_input("📍 X AutoCAD référence (mm)", value=0.0, step=1.0)
        ref_y_autocad = st.number_input("📍 Y AutoCAD référence (mm)", value=0.0, step=1.0)
    
    with col2:
        connect_points = st.checkbox("🔗 Relier les points", value=True)
        color_option = st.selectbox("🎨 Couleur", [("Rouge", 1), ("Jaune", 2), ("Vert", 3), ("Cyan", 4), ("Bleu", 5), ("Magenta", 6)], format_func=lambda x: x[0])
        color_code = color_option[1]
    
    if st.button("🚀 Générer DXF Final", type="primary"):
        if not dxf_file:
            st.error("⚠️ Veuillez uploader un plan DXF.")
            return
        
        try:
            # Création du fichier temporaire d'entrée
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                tmp.write(dxf_file.read())
                tmp_path = tmp.name
            
            # Lecture du DXF
            doc = ezdxf.readfile(tmp_path)
            msp = doc.modelspace()
            
            # Générateur DXF
            dxf_gen = DXFGenerator(cross_size, text_height)
            
            # Conversion et ajout des éléments
            positions_mm = relative_positions * 1000
            points_autocad = []
            
            for i, pos in enumerate(positions_mm):
                x = -pos[0] if inverse_x else pos[0]
                y = pos[2]  # Z → Y pour AutoCAD
                x_autocad = ref_x_autocad + x
                y_autocad = ref_y_autocad + y
                
                points_autocad.append((x_autocad, y_autocad))
                
                # Ajout croix et label
                dxf_gen.add_cross_and_label(msp, x_autocad, y_autocad, df.loc[i,'Nom'], color_code)
            
            # Lignes de connexion
            if connect_points and len(points_autocad) > 1:
                dxf_gen.add_connecting_lines(msp, points_autocad, color_code)
            
            # Sauvegarde
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as output_tmp:
                output_path = output_tmp.name
                doc.saveas(output_path)
            
            # Téléchargement
            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Télécharger le DXF Final",
                    data=f.read(),
                    file_name="plan_avec_bases.dxf",
                    mime="application/dxf"
                )
            
            # Nettoyage
            try:
                os.unlink(tmp_path)
                os.unlink(output_path)
            except:
                pass
                
            st.success("✅ DXF généré avec succès !")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération DXF: {str(e)}")
            logger.error(f"Erreur génération DXF: {e}")

if __name__ == "__main__":
    main()