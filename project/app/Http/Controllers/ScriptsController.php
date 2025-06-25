<?php

namespace App\Http\Controllers;

use App\Models\Scripts;
use Illuminate\Http\Request;

class ScriptsController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    
    public function index()
    {
        //
    }

    /**
     * Show the form for creating a new resource.
     */
    public function create()
    {
        //
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        //
    }

    /**
     * Display the specified resource.
     */
    public function show(Scripts $scripts)
    {
        //
    }

    /**
     * Show the form for editing the specified resource.
     */
    public function edit(Scripts $scripts)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, Scripts $scripts)
    {
        //
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(Scripts $scripts)
    {
        //
    }
public function saveOrUpdate(Request $request)
{
    $validated = $request->validate([
        'script_id' => 'required|string',
        'topic' => 'required|string',
        'raw_script' => 'required|string',
        'parsed_script' => 'required|array',
    ]);

    $exists = DB::connection('mysql_remote')->table('scripts')->where('script_id', $validated['script_id'])->exists();

    if ($exists) {
        DB::connection('mysql_remote')->table('scripts')->where('script_id', $validated['script_id'])->update([
            'topic' => $validated['topic'],
            'raw_script' => $validated['raw_script'],
            'parsed_script' => json_encode($validated['parsed_script']),
            'updated_at' => now(),
        ]);
    } else {
        DB::connection('mysql_remote')->table('scripts')->insert([
            'script_id' => $validated['script_id'],
            'topic' => $validated['topic'],
            'raw_script' => $validated['raw_script'],
            'parsed_script' => json_encode($validated['parsed_script']),
            'created_at' => now(),
            'updated_at' => now(),
        ]);
    }

    return response()->json(['success' => true, 'message' => 'Script saved to remote DB.']);
}

}
