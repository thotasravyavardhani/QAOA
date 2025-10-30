import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

export default function QuantumCircuitVisual({ numQubits, pLayers, problemType }) {
  // Generate a visual representation of the QAOA circuit
  const renderCircuit = () => {
    const qubits = Array.from({ length: numQubits }, (_, i) => i);
    
    return (
      <div className="bg-white p-6 rounded-lg border-2 border-indigo-200 overflow-x-auto">
        <div className="min-w-max">
          {/* Qubit lines */}
          {qubits.map((q) => (
            <div key={q} className="flex items-center mb-4 h-12">
              {/* Qubit label */}
              <div className="w-16 text-sm font-mono font-semibold text-gray-700">
                q[{q}]
              </div>
              
              {/* Initial state */}
              <div className="flex items-center">
                <div className="w-12 h-10 bg-blue-100 border-2 border-blue-500 rounded flex items-center justify-center text-xs font-semibold text-blue-700">
                  |0⟩
                </div>
                
                {/* Hadamard gate */}
                <div className="w-16 h-1 bg-gray-800"></div>
                <div className="w-10 h-10 bg-green-100 border-2 border-green-600 rounded flex items-center justify-center text-sm font-bold text-green-700">
                  H
                </div>
                
                {/* QAOA layers */}
                {Array.from({ length: pLayers }).map((_, layer) => (
                  <React.Fragment key={layer}>
                    {/* Cost Hamiltonian */}
                    <div className="w-16 h-1 bg-gray-800"></div>
                    <div className="w-12 h-10 bg-purple-100 border-2 border-purple-600 rounded flex items-center justify-center text-xs font-bold text-purple-700">
                      U_C
                    </div>
                    
                    {/* Mixer Hamiltonian */}
                    <div className="w-16 h-1 bg-gray-800"></div>
                    <div className="w-12 h-10 bg-orange-100 border-2 border-orange-600 rounded flex items-center justify-center text-xs font-bold text-orange-700">
                      U_M
                    </div>
                  </React.Fragment>
                ))}
                
                {/* Measurement */}
                <div className="w-16 h-1 bg-gray-800"></div>
                <div className="w-16 h-10 bg-red-100 border-2 border-red-500 rounded flex items-center justify-center">
                  <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2L12 12L17 7" className="text-red-600"/>
                    <circle cx="12" cy="12" r="8" className="text-red-600"/>
                  </svg>
                </div>
              </div>
            </div>
          ))}
          
          {/* Legend */}
          <div className="mt-6 pt-4 border-t border-gray-300">
            <div className="flex flex-wrap gap-4 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-green-100 border-2 border-green-600 rounded flex items-center justify-center font-bold text-green-700">H</div>
                <span className="text-gray-700">Hadamard (Superposition)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-purple-100 border-2 border-purple-600 rounded flex items-center justify-center text-xs font-bold text-purple-700">U_C</div>
                <span className="text-gray-700">Cost Hamiltonian (γ)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-orange-100 border-2 border-orange-600 rounded flex items-center justify-center text-xs font-bold text-orange-700">U_M</div>
                <span className="text-gray-700">Mixer Hamiltonian (β)</span>
              </div>
              <div className="flex items-center gap-2">
                <svg className="w-8 h-8 bg-red-100 border-2 border-red-500 rounded p-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 2L12 12L17 7" className="text-red-600"/>
                  <circle cx="12" cy="12" r="8" className="text-red-600"/>
                </svg>
                <span className="text-gray-700">Measurement</span>
              </div>
            </div>
            <div className="mt-3 text-xs text-gray-600 font-mono bg-gray-50 p-2 rounded">
              <strong>QAOA Circuit:</strong> p = {pLayers} layers | {numQubits} qubits | Depth ≈ {2 * pLayers + 1}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <Card className="mt-4">
      <CardHeader className="bg-gradient-to-r from-indigo-50 to-purple-50">
        <CardTitle className="flex items-center gap-2 text-indigo-900">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2L2 7L12 12L22 7L12 2Z"/>
            <path d="M2 17L12 22L22 17"/>
            <path d="M2 12L12 17L22 12"/>
          </svg>
          Quantum Circuit Architecture
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-6">
        {renderCircuit()}
      </CardContent>
    </Card>
  );
}
