#!/usr/bin/env bash
# SLHA v2 — One-click installer
# Usage: curl -sSL https://raw.githubusercontent.com/CHECKUPAUTO/SLHAv2/master/install.sh | bash

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

banner() {
    echo -e "${CYAN}"
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║       SLHA v2 — Installeur rapide       ║"
    echo "  ║   Faites tourner une IA sans GPU         ║"
    echo "  ╚══════════════════════════════════════════╝"
    echo -e "${NC}"
}

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERREUR]${NC} $*"; }

banner

# ── 1. Vérifier ou installer Rust ──────────────────────────────────
if ! command -v rustc &>/dev/null; then
    info "Rust n'est pas installé — installation en cours..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    info "Rust installé : $(rustc --version)"
else
    info "Rust détecté : $(rustc --version)"
fi

# ── 2. Cloner le dépôt ─────────────────────────────────────────────
REPO="https://github.com/CHECKUPAUTO/SLHAv2.git"
DIR="SLHAv2"

if [ -d "$DIR" ]; then
    warn "Le dossier '$DIR' existe déjà."
    read -rp "Le supprimer et re-cloner ? [o/N] " answer
    if [ "$answer" = "o" ] || [ "$answer" = "O" ]; then
        rm -rf "$DIR"
    else
        cd "$DIR"
        info "Utilisation du dossier existant."
    fi
fi

if [ ! -d "$DIR" ]; then
    info "Clonage de $REPO..."
    git clone "$REPO" "$DIR"
fi

cd "$DIR"

# ── 3. Compiler ─────────────────────────────────────────────────────
info "Compilation en mode release (2-3 minutes)..."
cargo build --release

# ── 4. Lancer les tests ─────────────────────────────────────────────
info "Lancement des tests..."
cargo test 2>&1 | tail -5

# ── 5. Premier essai ────────────────────────────────────────────────
echo ""
info "SLHA v2 est installé et prêt !"
echo ""
echo -e "  ${CYAN}Commandes utiles :${NC}"
echo "  cargo test                                  # Lancer tous les tests"
echo "  cargo run --example measure --release       # Benchmark complet"
echo "  cargo run --example basic_usage             # Exemple simple"
echo "  cargo bench                                 # Micro-benchmarks"
echo ""
echo -e "  ${CYAN}Documentation :${NC}"
echo "  docs/GETTING_STARTED.md    # Guide débutant"
echo "  docs/INTEGRATION.md        # Intégrer dans un projet"
echo "  SLHAv2.md                  # Spécification complète"
echo ""

# ── 6. Lancer l'exemple ─────────────────────────────────────────────
read -rp "Lancer l'exemple maintenant ? [O/n] " run
if [ "$run" != "n" ] && [ "$run" != "N" ]; then
    echo ""
    info "Lancement de l'exemple..."
    cargo run --example basic_usage
fi
