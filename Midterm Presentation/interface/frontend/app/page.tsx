"use client";

import Image from "next/image";
import { useState } from "react";

const backend_methods = {
    "Exhaustive Search": "exhaustive",
    "VPTree Search": "vp_tree",
    "Ball Tree Search": "ball_tree",
    "Spotify Annoy Search": "annoy",
    "Scipy cdist": "cdist",
}

const methods = Object.keys(backend_methods);

interface Result_Image {
    image_name: string;
    distance: number;
}

interface Result {
    status: number;
    results: Result_Image[];
    method: string;
    precision: number;
    recall: number;
    time: number;
}

export default function Home() {
    const [selectedMethod, setSelectedMethod] = useState(methods[0]);
    const [uploadedImage, setUploadedImage] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [results, setResults] = useState<Result>({
        status: 0,
        results: [],
        method: "",
        precision: 0,
        recall: 0,
        time: 0,
    });

    const handleMethodChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        setSelectedMethod(event.target.value);
    };

    const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setUploadedImage(file);
            setPreviewUrl(URL.createObjectURL(file));
        }
    };

    const handleFormSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!uploadedImage) {
            alert("Please upload an image before submitting.");
            return;
        }

        const formData = new FormData();
        formData.append("method", backend_methods[selectedMethod as keyof typeof backend_methods]);
        formData.append("file", uploadedImage);

        try {
            const res = await fetch("http://localhost:8000/query", {
                method: "POST",
                body: formData,
            });

            const data = await res.json();
            setResults(data);
        } catch (error) {
            console.error("Error submitting the form:", error);
        }
    };

    return (
        <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
            <main className="flex min-h-screen w-full flex-col items-center py-24 px-16 bg-white dark:bg-black gap-8">
                <div className="text-center">
                    <h1 className="font-semibold text-2xl">CBIR Querying Interface</h1>
                    <h2>Choose a method and upload an image</h2>
                </div>

                <form onSubmit={handleFormSubmit} className="flex flex-col items-center gap-4">
                    <label className="flex flex-row gap-2 items-baseline">
                        <p>Search Method:</p>
                        <select
                            className="border border-zinc-300 rounded-lg p-2 w-48 mb-4"
                            value={selectedMethod}
                            onChange={handleMethodChange}
                        >
                            {methods.map((method) => (
                                <option key={method}>{method}</option>
                            ))}
                        </select>
                    </label>
                    <label>
                        <input type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
                        <div className="border-2 border-dashed border-zinc-300 rounded-lg w-80 h-80 flex flex-col items-center justify-center cursor-pointer hover:border-zinc-500">
                            {previewUrl ? (
                                <Image src={previewUrl} alt="Uploaded Preview" width={256} height={256} />
                            ) : (
                                <>
                                    <Image src="/upload_icon.png" alt="Upload Icon" width={256} height={256} />
                                    <span className="mt-4 text-zinc-500">Click to upload an image</span>
                                </>
                            )}
                        </div>
                    </label>

                    {uploadedImage && <p className="text-green-600">{uploadedImage.name} uploaded!</p>}
                    <button
                        type="submit"
                        className="mt-4 cursor-pointer bg-blue-400 text-white px-6 py-2 rounded-lg hover:bg-blue-600"
                    >
                        Submit
                    </button>
                </form>

                {results.results.length > 0 && (
                    <div className="w-full">
                        <h2 className="text-center font-semibold text-xl mb-4">Results:</h2>
                        <div>
                            <h3>Precision: {results.precision}</h3>
                            <h3>Recall: {results.recall}</h3>
                            <h3>Time Taken: {results.time * 1e6} microseconds</h3>
                        </div>
                        <div className="grid grid-cols-4 gap-4">
                            {results.results.map((result, index) => (
                                <div key={index} className="flex flex-col items-center">
                                    <img src={'http://localhost:8000/' + result.image_name} alt={`Result ${index + 1}`} className="w-48 h-48 object-cover rounded-lg" />
                                    <p className="mt-2 text-sm">Distance: {result.distance.toFixed(4)}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}
