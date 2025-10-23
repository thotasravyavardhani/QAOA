import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Construction } from 'lucide-react';

export default function VRPSolver() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Vehicle Routing Problem (VRP)</CardTitle>
        <CardDescription>Optimize routes for multiple vehicles</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-center py-12 text-gray-400">
          <Construction className="w-16 h-16 mx-auto mb-4" />
          <p>VRP Solver - Similar to Max-Cut implementation</p>
          <p className="text-sm mt-2">Configure customers, vehicles, and capacity constraints</p>
        </div>
      </CardContent>
    </Card>
  );
}
