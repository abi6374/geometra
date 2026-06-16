import { Suspense, useRef } from "react";
import { Canvas, useFrame, type ThreeEvent } from "@react-three/fiber";
import {
  OrbitControls,
  Grid,
  Environment,
  ContactShadows,
  Html,
  useGLTF,
} from "@react-three/drei";
import * as THREE from "three";

interface Viewer3DProps {
  modelPath?: string;
  /** If true, shows an interactive viewer; otherwise a placeholder */
  hasModel?: boolean;
}

function Model({ path }: { path: string }) {
  const { scene } = useGLTF(path);
  return <primitive object={scene} scale={1} />;
}

function PlaceholderModel() {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.getElapsedTime() * 0.3;
    }
  });

  return (
    <mesh ref={meshRef} onClick={(e: ThreeEvent<MouseEvent>) => e.stopPropagation()}>
      <boxGeometry args={[2, 1.5, 1]} />
      <meshStandardMaterial color="#4f46e5" wireframe={false} roughness={0.3} metalness={0.1} />
      <Edges />
    </mesh>
  );
}

function Edges() {
  return (
    <mesh>
      <boxGeometry args={[2.01, 1.51, 1.01]} />
      <meshBasicMaterial color="#818cf8" wireframe transparent opacity={0.3} />
    </mesh>
  );
}

function LoadingFallback() {
  return (
    <Html center>
      <div className="flex items-center gap-2 text-slate-400">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-indigo-400 border-t-transparent" />
        <span>Loading 3D model...</span>
      </div>
    </Html>
  );
}

export function Viewer3D({ modelPath, hasModel = false }: Viewer3DProps) {
  return (
    <div className="h-full w-full rounded-lg bg-gradient-to-b from-slate-900 to-slate-950">
      <Canvas
        shadows
        camera={{ position: [5, 4, 5], fov: 45 }}
        gl={{ antialias: true, alpha: false }}
        onCreated={({ gl }) => {
          gl.setClearColor(new THREE.Color("#0f172a"));
        }}
      >
        <Suspense fallback={<LoadingFallback />}>
          <ambientLight intensity={0.4} />
          <directionalLight
            position={[5, 10, 5]}
            intensity={1.2}
            castShadow
            shadow-mapSize-width={1024}
            shadow-mapSize-height={1024}
          />
          <directionalLight position={[-3, 5, -3]} intensity={0.5} />
          <pointLight position={[0, 3, 0]} intensity={0.3} />

          {hasModel && modelPath ? (
            <Model path={modelPath} />
          ) : (
            <PlaceholderModel />
          )}

          <Grid
            position={[0, -0.8, 0]}
            args={[10, 10]}
            cellColor="#334155"
            sectionColor="#475569"
            cellSize={0.5}
            sectionSize={1}
          />
          <ContactShadows
            position={[0, -0.79, 0]}
            opacity={0.4}
            scale={8}
            blur={2.5}
          />
          <OrbitControls
            enableDamping
            dampingFactor={0.08}
            minDistance={2}
            maxDistance={20}
          />
          <Environment preset="studio" />
        </Suspense>
      </Canvas>
    </div>
  );
}
