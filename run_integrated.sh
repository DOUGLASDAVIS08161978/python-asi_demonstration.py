#!/usr/bin/env bash
# Integrated deployment and execution script for ASI Demonstration Systems

set -e

echo "================================================================================"
echo "🚀 ASI DEMONSTRATION SYSTEMS - INTEGRATED DEPLOYMENT"
echo "================================================================================"
echo ""

# Check if required executables exist
check_requirements() {
    echo "📋 Checking system requirements..."
    
    if ! command -v python3 &> /dev/null; then
        echo "❌ Error: python3 is not installed"
        exit 1
    fi
    
    if ! command -v node &> /dev/null; then
        echo "❌ Error: node is not installed"
        exit 1
    fi
    
    echo "✅ Python3: $(python3 --version)"
    echo "✅ Node.js: $(node --version)"
    echo ""
}

# Run Python ASI Demonstration
run_python_asi() {
    echo "================================================================================"
    echo "🐍 PYTHON ASI DEMONSTRATION SYSTEM"
    echo "================================================================================"
    echo ""
    
    python3 asi_demonstration.py
    
    echo ""
}

# Run Node.js ARIA System
run_nodejs_aria() {
    echo "================================================================================"
    echo "🌌 NODE.JS ARIA QUANTUM METACOGNITION SYSTEM"
    echo "================================================================================"
    echo ""
    
    node aria.js
    
    echo ""
}

# Main execution
main() {
    check_requirements
    
    echo "Starting integrated demonstration..."
    echo ""
    
    # Run Python system first
    run_python_asi
    
    # Brief pause between systems
    sleep 2
    
    # Run Node.js system
    run_nodejs_aria
    
    echo "================================================================================"
    echo "🎉 INTEGRATED DEMONSTRATION COMPLETE"
    echo "================================================================================"
    echo ""
    echo "Both ASI demonstration systems executed successfully!"
    echo ""
    echo "Systems demonstrated:"
    echo "  ✓ Python ASI: Self-modification, meta-learning, consciousness modeling"
    echo "  ✓ Node.js ARIA: Quantum metacognition, multiversal optimization"
    echo ""
}

# Execute main function
main
